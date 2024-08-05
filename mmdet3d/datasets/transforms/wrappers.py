# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Union

from mmcv.transforms import BaseTransform, Compose

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiBranch3D(BaseTransform):
    r"""Multiple branch pipeline wrapper.

    Args:
        branch_field (list): List of branch names.
        branch_pipelines (dict): Dict of different pipeline configs to be
            composed.
    """

    def __init__(self, branch_field: List[str],
                 **branch_pipelines: dict) -> None:
        self.branch_field = branch_field
        self.branch_pipelines = {
            branch: Compose(pipeline)
            for branch, pipeline in branch_pipelines.items()
        }

    def transform(self, results: dict) -> dict:
        """Transform function to apply transforms sequentially.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict:

            - 'inputs' (Dict[dict]): The forward data of models from different
              branches.
            - 'data_samples' (Dict[str, :obj:`Det3DDataSample`]): The
              annotation info of the sample different branches.
        """
        multi_results = {}
        for branch, pipeline in self.branch_pipelines.items():
            branch_results = pipeline(copy.deepcopy(results))
            # If one branch pipeline returns None,
            # it will sample another data from dataset.
            if branch_results is None:
                return None
            multi_results[branch] = branch_results

        branch_fields = list(self.branch_pipelines.keys())
        for branch in self.branch_field:
            if multi_results.get(branch, None) is None:
                multi_results[branch] = {
                    'inputs': dict(),
                    'data_samples': None
                }
                for key in multi_results[branch_fields[0]]['inputs'].keys():
                    multi_results[branch]['inputs'][key] = None

        format_results = {}
        for branch, results in multi_results.items():
            for key in results.keys():
                if format_results.get(key, None) is None:
                    format_results[key] = {branch: results[key]}
                else:
                    format_results[key][branch] = results[key]
        return format_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(branch_pipelines={list(self.branch_pipelines.keys())})'
        return repr_str



@TRANSFORMS.register_module()
class SplitTransform(BaseTransform):
    """Apply transforms randomly with a given probability.

    Args:
        transforms (list[dict | callable]): The transform or transform list
            to randomly apply.
        prob (float): The probability to apply transforms. Default: 0.5

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomApply',
        >>>         transforms=[dict(type='HorizontalFlip')],
        >>>         prob=0.3)
        >>> ]
    """

    def __init__(self,
                 transforms: Union[BaseTransform, List[BaseTransform]]):

        super().__init__()
        self.transforms = Compose(transforms)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly apply the transform."""
        results_original = {}
        results_aug = {}
        for key, value in results.items():
            results_original[key] = value
            results_aug[key] = value
        results_aug = self.transforms(results_aug)

        results = {**results_original}
        results['points_aug'] = results_aug['points']
        results['pts_semantic_mask_aug'] = results_aug['pts_semantic_mask']
        return results


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f', prob = {self.prob})'
        return repr_str
