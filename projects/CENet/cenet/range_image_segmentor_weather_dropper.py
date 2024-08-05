# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from torch import Tensor

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList

## for DQN
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.utils import rename_loss_dict
import random
import numpy as np

import math
from collections import namedtuple, deque
from mmdet3d.structures import PointData
from typing import Dict
from mmdet3d.registry import MODELS


DQN_DEFAULTS = {
    'train_start_iter': 32,
    'gamma': 0.99,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 1000,
    'tau': 0.005,
    'replay_memory_size': 10000,
    'drop_threshold': 0.5,
    'drop_threshold_explore': 0.5
}

RANGELIZER_DEFAULTS = {
    'H': 64,
    'W': 512,
    'fov_up': 3.0,
    'fov_down': -25.0,
    'means': (11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
    'stds': (10.24, 12.295865, 9.4287, 0.8643, 0.1450),
    'ignore_index': 19
}

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition saving"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    '''
    https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    def __init__(self, n_observations=4, n_actions=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, state, points):
        loss_ = state[:, 0]
        uncertainty_ = state[:, 1]

        x_ = loss_ + uncertainty_ + points

        x = F.relu(self.fc1(x_))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        out = F.softmax(out, dim=0) ## take threshold for True/False
        return out
    

@MODELS.register_module()
class RangeImageSegmentor_WeatherDropper(EncoderDecoder3D):
    def __init__(self,
                 n_observations=4,
                 n_actions=1,
                 **kwargs) -> None:
        super(RangeImageSegmentor_WeatherDropper, self).__init__(**kwargs)

        self.learnable_drop = self.train_cfg.get('learnable_drop', False)
        self.cache = dict()

        # Initialize Dropper parameters
        self.init_dqn_parameters()

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.freeze(self.target_net)
        self.freeze(self.policy_net)

        self.memory = ReplayMemory(self.replay_memory_size)
        self.steps_done = 0
        self.next_policy = None

        # Initialize Rangelizer
        self.init_rangelizer()

    def init_dqn_parameters(self):
        self.BATCH_SIZE = self.train_cfg.get('dqn_train_start_iter', DQN_DEFAULTS['train_start_iter'])
        self.GAMMA = self.train_cfg.get('dqn_gamma', DQN_DEFAULTS['gamma'])
        self.EPS_START = self.train_cfg.get('dqn_eps_start', DQN_DEFAULTS['eps_start'])
        self.EPS_END = self.train_cfg.get('dqn_eps_end', DQN_DEFAULTS['eps_end'])
        self.EPS_DECAY = self.train_cfg.get('dqn_eps_decay', DQN_DEFAULTS['eps_decay'])
        self.TAU = self.train_cfg.get('dqn_tau', DQN_DEFAULTS['tau'])
        self.replay_memory_size = self.train_cfg.get('dqn_replay_memory_size', DQN_DEFAULTS['replay_memory_size'])
        self.drop_threshold = self.train_cfg.get('dqn_drop_threshold', DQN_DEFAULTS['drop_threshold'])
        self.drop_threshold_explore = self.train_cfg.get('dqn_drop_threshold_explore', DQN_DEFAULTS['drop_threshold_explore'])

    def init_rangelizer(self):
        H = self.train_cfg.get('rangelizer_H', RANGELIZER_DEFAULTS['H'])
        W = self.train_cfg.get('rangelizer_W', RANGELIZER_DEFAULTS['W'])
        fov_up = self.train_cfg.get('rangelizer_fov_up', RANGELIZER_DEFAULTS['fov_up'])
        fov_down = self.train_cfg.get('rangelizer_fov_down', RANGELIZER_DEFAULTS['fov_down'])
        means = self.train_cfg.get('rangelizer_means', RANGELIZER_DEFAULTS['means'])
        stds = self.train_cfg.get('rangelizer_stds', RANGELIZER_DEFAULTS['stds'])
        ignore_index = self.train_cfg.get('rangelizer_ignore_index', RANGELIZER_DEFAULTS['ignore_index'])
        self.rangelizer = PseudoSemkittiRangeView(
            H=H,
            W=W,
            fov_up=fov_up,
            fov_down=fov_down,
            means=means,
            stds=stds,
            ignore_index=ignore_index
        )

    @staticmethod
    def freeze(model: nn.Module) -> None:
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model: nn.Module) -> None:
        """Freeze the model."""
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    def calculate_reward(self, last_loss_unc, current_loss_unc):
        """
        Implement the logic to calculate the reward based on the increase of the loss and uncertainty
        """
        mean_increase = current_loss_unc - last_loss_unc
        return mean_increase

    def select_action(self, state, points):
        """Return the point index to drop based on epsilon-greedy strategy."""
        # global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                prob = self.policy_net(state, points)
                return (prob < self.drop_threshold) # drop higher than 0.5
        else:
            out = torch.rand(points.shape[0], device=points.device)
            return (out < self.drop_threshold_explore) # drop higher than 0.5
            
    @torch.no_grad()
    def get_drop_points(self, cache):
        """
        need to be fixed for range image version
        """
        loss = cache['loss']
        uncertainty = cache['uncertainty']
        batch_inputs = cache['batch_inputs']
        batch_data_samples = cache['batch_data_samples']

        enum_list = []
        mask_list = []
        points_list = []
        action_list = []
        for i in range(len(batch_inputs['points'])):
            enum = torch.tensor([i]*len(batch_inputs['points'][i]))
            mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            points = batch_inputs['points'][i]
            state = torch.tensor([loss, uncertainty], device=points.device).reshape(1, -1)
            action = self.select_action(state, points).reshape(-1,)
            self.next_policy = action

            enum_list.append(enum[action]) ## drop
            mask_list.append(mask[action]) ## drop
            points_list.append(points[action]) ## drop
            action_list.append(action) ## drop
        enum = torch.cat(enum_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        points = torch.cat(points_list, dim=0)
        action = torch.cat(action_list, dim=0)

        point_results = {
            'points' : [],
            'pts_semantic_mask' : [],
        }
        for i in range(len(batch_inputs['points'])):
            points_idx = (enum == i)
            point_results['points'].append(points[points_idx])
            point_results['pts_semantic_mask'].append(mask[points_idx])

        ## rangelize points
        batch_inputs_out, batch_data_samples_out = self.rangelizer.transform(point_results, batch_data_samples)
        return batch_inputs_out, batch_data_samples_out, action, state
    

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        # extract features using backbone
        imgs = batch_inputs_dict['imgs']
        x = self.extract_feat(imgs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, batch_data_samples)
            losses.update(loss_aux)

        ## Learnabnle Point Drop loss trick
        if self.learnable_drop:
            with torch.no_grad():
                range_feature = self.extract_feat(batch_inputs_dict['imgs'])
                logits_softmax = F.softmax(self.decode_head.forward(range_feature), dim=1)
                self.cache = {
                    'batch_inputs' : batch_inputs_dict,
                    'batch_data_samples' : batch_data_samples,
                    'loss' : loss_decode['decode.loss_ce'],
                    'uncertainty' : torch.mean(torch.sum(-logits_softmax*torch.log(logits_softmax + 1e-8), dim=1)),
                }
                batch_inputs_dict_drop, batch_data_samples_drop, action, state = self.get_drop_points(self.cache)

            range_feature_drop = self.extract_feat(batch_inputs_dict_drop['imgs'])
            loss_drop = self._decode_head_forward_train(range_feature_drop, batch_data_samples_drop)
            loss_drop = rename_loss_dict('drop_', loss_drop)
            losses.update(loss_drop)
            with torch.no_grad():
                logits_softmax_drop = F.softmax(self.decode_head.forward(range_feature_drop), dim=1)
                uncertainty_drop = torch.mean(torch.sum(-logits_softmax_drop*torch.log(logits_softmax_drop + 1e-8), dim=1))
                next_state = torch.tensor([loss_drop['drop_decode.loss_ce'], uncertainty_drop], device=batch_inputs_dict['points'][0].device).reshape(1, -1)
                reward = self.calculate_reward(loss_decode['decode.loss_ce'], loss_drop['drop_decode.loss_ce'])
                self.memory.push(state, action, next_state, reward) # Update memory

            # Policy net update
            if len(self.memory) > self.BATCH_SIZE:
                loss_policy = self.update_policy_net(batch_inputs_dict)
                losses.update(loss_policy)
        return losses

    def update_policy_net(self, batch_inputs_dict):
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device=batch_inputs_dict['points'][0].device)
        state_batch = torch.cat(batch.state)
        reward_batch = torch.stack(batch.reward)

        state_action_values_list = []
        expected_state_action_values_list = []

        for i in range(len(batch_inputs_dict['points'])):
            state_action_values = self.policy_net(state_batch[i].reshape(1, -1), batch_inputs_dict['points'][i])
            with torch.no_grad():
                next_state_values = self.target_net(non_final_next_states[i].reshape(1, -1), batch_inputs_dict['points'][i])
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch[i]
            state_action_values_list.append(state_action_values)
            expected_state_action_values_list.append(expected_state_action_values)

        state_action_values = torch.cat(state_action_values_list, dim=0)
        expected_state_action_values = torch.cat(expected_state_action_values_list, dim=0)

        loss_policy = {'dqn_loss': F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())}
        return loss_policy

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        imgs = batch_inputs_dict['imgs']
        x = self.extract_feat(imgs)
        seg_labels_list = self.decode_head.predict(x, batch_input_metas,
                                                   self.test_cfg)

        return self.postprocess_result(seg_labels_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        imgs = batch_inputs_dict['imgs']
        x = self.extract_feat(imgs)
        return self.decode_head.forward(x)

    def postprocess_result(self, seg_labels_list: List[Tensor],
                           batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_labels_list (List[Tensor]): List of segmentation results,
                seg_logits from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """

        for i, seg_pred in enumerate(seg_labels_list):
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': seg_pred})})
        return batch_data_samples


from typing import Sequence
import numpy as np
from mmcv.transforms import BaseTransform
class PseudoSemkittiRangeView(BaseTransform):
    """
    Convert Semantickitti point cloud dataset to range image.
    only need for training CENet.
    """

    def __init__(self,
                 H: int = 64,
                 W: int = 512,
                 fov_up: float = 3.0,
                 fov_down: float = -25.0,
                 means: Sequence[float] = (11.71279, -0.1023471, 0.4952,
                                           -1.0545, 0.2877),
                 stds: Sequence[float] = (10.24, 12.295865, 9.4287, 0.8643,
                                          0.1450),
                 ignore_index: int = 19) -> None:

        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.ignore_index = ignore_index
    
    @torch.no_grad()
    def transform(self, results: dict, batch_data_samples) -> dict:
        """
        batch_data_samples_out = copy.deepcopy(batch_data_samples)
        for i in range(len(batch_inputs['points'])):
            points_idx = (enum == i)
            batch_inputs_out['points'].append(points[points_idx]) # batch_inputs_out['points'][i] = points[points_idx]
            batch_data_samples_out[i].set_data({'gt_pts_seg':PointData(**{'pts_semantic_mask': mask[points_idx]})})
        batch_inputs_out['voxels'] = self.data_preprocessor.voxelize(batch_inputs_out['points'], batch_data_samples_out)
        """
        batch_inputs_out = dict()
        batch_inputs_out['points'] = []
        batch_inputs_out['imgs'] = []
        batch_data_samples_out = copy.deepcopy(batch_data_samples)
        
        for i in range(len(results['points'])):
            points_numpy = results['points'][i].cpu().detach().numpy()

            proj_image = np.full((self.H, self.W, 5), -1, dtype=np.float32)
            proj_idx = np.full((self.H, self.W), -1, dtype=np.int64)

            # get depth of all points
            depth = np.linalg.norm(points_numpy[:, :3], 2, axis=1)

            # get angles of all points
            yaw = -np.arctan2(points_numpy[:, 1], points_numpy[:, 0])
            pitch = np.arcsin(points_numpy[:, 2] / depth)

            # get projection in image coords
            proj_x = 0.5 * (yaw / np.pi + 1.0)
            proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            # scale to image size using angular resolution
            proj_x *= self.W
            proj_y *= self.H

            # round and clamp for use as index
            proj_x = np.floor(proj_x)
            proj_x = np.minimum(self.W - 1, proj_x)
            proj_x = np.maximum(0, proj_x).astype(np.int64)

            proj_y = np.floor(proj_y)
            proj_y = np.minimum(self.H - 1, proj_y)
            proj_y = np.maximum(0, proj_y).astype(np.int64)

            # order in decreasing depth
            indices = np.arange(depth.shape[0])
            order = np.argsort(depth)[::-1]
            proj_idx[proj_y[order], proj_x[order]] = indices[order]
            proj_image[proj_y[order], proj_x[order], 0] = depth[order]
            proj_image[proj_y[order], proj_x[order], 1:] = points_numpy[order]
            proj_mask = (proj_idx > 0).astype(np.int32)

            proj_image = (proj_image -
                        self.means[None, None, :]) / self.stds[None, None, :]
            proj_image = proj_image * proj_mask[..., None].astype(np.float32)
            batch_inputs_out['imgs'].append(proj_image.transpose((2,0,1))) # HWC -> CHW
            batch_inputs_out['points'].append(results['points'][i])
            
            if 'pts_semantic_mask' in results:
                proj_sem_label = np.full((self.H, self.W),
                                        self.ignore_index,
                                        dtype=np.int64)
                proj_sem_label[proj_y[order],
                            proj_x[order]] = results['pts_semantic_mask'][i].cpu().detach().numpy()[order]
                proj_sem_label = torch.tensor(proj_sem_label, dtype=torch.int64, device=results['points'][i].device)
                batch_data_samples_out[i].set_data({'gt_pts_seg':PointData(**{'semantic_seg': proj_sem_label, 'pts_semantic_mask': batch_data_samples_out[i].gt_pts_seg.pts_semantic_mask})})

        ##### merge image samples
        batch_inputs_out['imgs'] = torch.tensor(np.stack(batch_inputs_out['imgs'], axis=0), dtype=torch.float32, device=batch_inputs_out['points'][0].device)
        return batch_inputs_out, batch_data_samples_out