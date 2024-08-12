# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, OrderedDict, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import Base3DSegmentor


@MODELS.register_module()
class MeanTeacher3DSegmentor(Base3DSegmentor):
    """Base class for semi-supervisied segmentors.

    Semi-supervisied segmentors typically consisting of a teacher model updated
    by exponential moving average and a student model updated by gradient
    descent.

    Args:
        segmentor (:obj:`ConfigDict` or dict): The segmentor config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The
            semi-supervised training config. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The semi-segmentor
            testing config. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`Det3DDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`ConfigDict` or dict],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 segmentor_student: ConfigType,
                 segmentor_teacher: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(MeanTeacher3DSegmentor, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(segmentor_student)
        self.teacher = MODELS.build(segmentor_teacher)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg.get('freeze_teacher', True) is True:
            self.freeze(self.teacher)

    @staticmethod
    def freeze(model: nn.Module) -> None:
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, batch_inputs: Dict[str, dict],
             batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        losses = dict()
        _, losses_src = self.loss_by_gt_instances(
            batch_inputs, batch_data_samples)
        losses.update(**losses_src)

        _, pseudo_data_samples = self.get_pseudo_instances(
            batch_inputs, batch_data_samples)
        _, losses_const = self.loss_by_pseudo_instances(
            batch_inputs, pseudo_data_samples)
        losses.update(**losses_const)
        return losses

    def loss_by_gt_instances(
            self, batch_inputs: dict,
            batch_data_samples: SampleList) -> Tuple[Tensor, dict]:
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (dict): Input sample dict which includes 'points' and
                'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
                
        Returns:
            Tuple[Tensor, dict]: Predict logits and a dictionary of loss
            components.
        """
        logits = self.student(batch_inputs, batch_data_samples, mode='tensor')
        sup_weight = self.train_cfg.get('main_weight', 1.)
        losses = self.student.decode_head.loss_by_feat(logits,
                                                    batch_data_samples)
        losses = rename_loss_dict('main_',
                                reweight_loss_dict(losses, sup_weight))
        return logits, losses

    def loss_by_pseudo_instances(self, batch_inputs: dict,
                                 batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (dict): Input sample dict which includes 'points' and
                'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        logits = self.student(batch_inputs, batch_data_samples, mode='tensor')
        sup_weight = self.train_cfg.get('pseudo_weight', 1.)
        losses = self.student.decode_head.loss_by_feat(logits,
                                                    batch_data_samples)
        losses = rename_loss_dict('pseudo_main_',
                                reweight_loss_dict(losses, sup_weight))
        return logits, losses

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tensor, SampleList]:
        """Get pseudo instances from teacher model."""
        logits = self.teacher(batch_inputs, batch_data_samples, mode='tensor')
        results_list = self.teacher.decode_head.predict_by_feat(
            logits, batch_data_samples)
        for i, (data_samples, results) in enumerate(zip(batch_data_samples, results_list)):
            seg_logits = F.softmax(results, dim=1)
            seg_scores, seg_labels = seg_logits.max(dim=1)
            pseudo_thr = self.train_cfg.get('pseudo_thr', 0.)
            ignore_mask = (seg_scores < pseudo_thr)
            seg_labels[ignore_mask] = self.train_cfg.ignore_label
            projection_mask = batch_inputs['projection_mask'][i] ## newly added
            seg_labels_pjt = seg_labels[projection_mask]
            data_samples.set_data(
                {'gt_pts_seg': PointData(**{'pts_semantic_mask': seg_labels, 'pts_semantic_mask_pjt': seg_labels_pjt})})
        return logits, batch_data_samples

    @torch.no_grad()
    def get_pseudo_instances_range_view(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tensor, SampleList]:
        """Get pseudo instances from teacher model."""
        logits = self.teacher(batch_inputs, batch_data_samples, mode='tensor')
        # results_list = self.teacher.decode_head.predict_by_feat(
        #     logits, batch_data_samples)
        for data_samples, logit in zip(batch_data_samples, logits):
            seg_logit = F.softmax(logit, dim=1)
            seg_score, seg_label = seg_logit.max(dim=1)  # [h, w]
            pseudo_thr = self.train_cfg.get('pseudo_thr', 0.)
            ignore_mask = (seg_score < pseudo_thr)
            seg_label[ignore_mask] = self.train_cfg.ignore_label
            data_samples.set_data(
                {'gt_pts_seg': PointData(**{'semantic_seg': torch.unsqueeze(seg_label, dim=0)})})
        return logits, batch_data_samples

    def _update_ema_variables(self, momentum):
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.mul_(1 - momentum).add_(param_s.data, alpha=momentum)

        ## do EMAN (https://arxiv.org/abs/2101.08482)
        # for buffer_t, buffer_s in zip(self.teacher.buffers(), self.student.buffers()):
        #     buffer_t.data.mul_(1 - momentum).add_(buffer_s.data, alpha=momentum)

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        if self.test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')

    def _forward(self,
                 batch_inputs: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        if self.test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='tensor')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> dict:
        if self.test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(batch_inputs)
        else:
            return self.student.extract_feat(batch_inputs)

    def encode_decode(self, batch_inputs: Tensor,
                      batch_data_samples: SampleList) -> Tensor:
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    def _load_from_state_dict(self, state_dict: OrderedDict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: List[str],
                              unexpected_keys: List[str],
                              error_msgs: List[str]) -> None:
        if not any([
                'student' in key or 'teacher' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(state_dict, prefix,
                                             local_metadata, strict,
                                             missing_keys, unexpected_keys,
                                             error_msgs)
