from mmengine.hooks import Hook
from mmengine.model.wrappers.distributed import MMDistributedDataParallel
from mmdet3d.registry import HOOKS
import torch

from typing import Optional, Union
DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class PolicyTargetHook(Hook):
    def __init__(self, dataset_name='semantickitti', train_start_iter=32):
        pass

    def before_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        if len(model.memory) > model.BATCH_SIZE:
            model.unfreeze(model.policy_net)


    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        with torch.no_grad():
            target_net_state_dict = model.target_net.state_dict()
            policy_net_state_dict = model.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*model.TAU + target_net_state_dict[key]*(1-model.TAU)
            model.target_net.load_state_dict(target_net_state_dict)