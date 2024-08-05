# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.utils import rename_loss_dict
import random
import os
import os.path as osp
import numpy as np

import math
from collections import namedtuple, deque
from mmdet3d.structures import PointData
from typing import Dict
from mmdet3d.registry import MODELS

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
class MinkUNetWeatherDropper(EncoderDecoder3D):
    r"""MinkUNetWeatherDropper is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self,
                 n_observations=4,
                 n_actions=1,
                 **kwargs) -> None:
        super(MinkUNetWeatherDropper, self).__init__(**kwargs)
        self.cache = dict()
        self.learnable_drop = self.train_cfg.get('learnable_drop', False)

        # Initialize DQN parameters
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

    def init_dqn_parameters(self):
        """Initialize DQN parameters from the configuration."""
        self.BATCH_SIZE = self.train_cfg.get('dqn_train_start_iter', 32)
        self.GAMMA = self.train_cfg.get('dqn_gamma', 0.99)
        self.EPS_START = self.train_cfg.get('dqn_eps_start', 0.9)
        self.EPS_END = self.train_cfg.get('dqn_eps_end', 0.05)
        self.EPS_DECAY = self.train_cfg.get('dqn_eps_decay', 1000)
        self.TAU = self.train_cfg.get('dqn_tau', 0.005)
        self.replay_memory_size = self.train_cfg.get('dqn_replay_memory_size', 10000)
        self.drop_threshold = self.train_cfg.get('dqn_drop_threshold', 0.5)
        self.drop_threshold_explore = self.train_cfg.get('dqn_drop_threshold_explore', 0.5)

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
        """Calculate the reward based on the increase of the loss and uncertainty."""
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
        loss = cache['loss']
        uncertainty = cache['uncertainty']
        batch_inputs = cache['batch_inputs']
        batch_data_samples = cache['batch_data_samples']

        enum_list, mask_list, points_list, action_list = [], [], [], []
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

        ## voxelize points
        batch_inputs_out = {'points': []}
        batch_data_samples_out = copy.deepcopy(batch_data_samples)
        for i in range(len(batch_inputs['points'])):
            points_idx = (enum == i)
            batch_inputs_out['points'].append(points[points_idx])
            batch_data_samples_out[i].set_data({'gt_pts_seg':PointData(**{'pts_semantic_mask': mask[points_idx]})})
        batch_inputs_out['voxels'] = self.data_preprocessor.voxelize(batch_inputs_out['points'], batch_data_samples_out)
        return batch_inputs_out, batch_data_samples_out, action, state

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            dict: The dict containing features.
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        voxel_dict['voxel_feats'] = x
        return voxel_dict

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
        losses = dict()

        voxel_dict = self.extract_feat(batch_inputs_dict)
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                    batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)

        if self.learnable_drop:
            voxel_dict = self.extract_feat(batch_inputs_dict)
            with torch.no_grad():
                logits_softmax = F.softmax(self.decode_head(voxel_dict)['logits'], dim=1)
                self.cache = {
                    'batch_inputs' : batch_inputs_dict,
                    'batch_data_samples' : batch_data_samples,
                    'loss' : loss_decode['decode.loss_ce'],
                    'uncertainty' : torch.mean(torch.sum(-logits_softmax*torch.log(logits_softmax + 1e-8), dim=1)),
                }
                batch_inputs_dict_drop, batch_data_samples_drop, action, state = self.get_drop_points(self.cache)
            voxel_dict_drop = self.extract_feat(batch_inputs_dict_drop)
            loss_drop = self._decode_head_forward_train(voxel_dict_drop, batch_data_samples_drop)
            loss_drop = rename_loss_dict('drop_', loss_drop)
            losses.update(loss_drop)
            with torch.no_grad():
                logits_softmax_drop = F.softmax(self.decode_head(voxel_dict_drop)['logits'], dim=1)
                uncertainty_drop = torch.mean(torch.sum(-logits_softmax_drop*torch.log(logits_softmax_drop + 1e-8), dim=1))
                next_state = torch.tensor([loss_drop['drop_decode.loss_ce'], uncertainty_drop], device=batch_inputs_dict['points'][0].device).reshape(1, -1)
                reward = self.calculate_reward(loss_decode['decode.loss_ce'], loss_drop['drop_decode.loss_ce'])
                self.memory.push(state, action, next_state, reward) # Update memory

            # Policy net update
            if len(self.memory) > self.BATCH_SIZE:
                loss_policy = self.update_policy_net(batch_inputs_dict)
                losses.update(loss_policy)
        return losses
    
    def update_policy_net(self, batch_inputs_dict: dict):
        """Update the policy network based on sampled transitions."""
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
        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
                                                   batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
        return self.postprocess_result(seg_logits_list, batch_data_samples)

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
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)