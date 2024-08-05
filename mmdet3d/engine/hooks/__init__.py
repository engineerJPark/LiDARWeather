# Copyright (c) OpenMMLab. All rights reserved.
from .benchmark_hook import BenchmarkHook
from .disable_object_sample_hook import DisableObjectSampleHook
from .visualization_hook import Det3DVisualizationHook
### newly added
from .policy_target_hook import PolicyTargetHook

__all__ = [
    'Det3DVisualizationHook', 'BenchmarkHook', 'DisableObjectSampleHook',
    ### newly added
    'PolicyTargetHook',
]
