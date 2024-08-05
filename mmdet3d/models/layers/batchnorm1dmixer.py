import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchsparse.nn.utils import fapply
from torchsparse import SparseTensor

class BatchNorm1dMixer(nn.BatchNorm1d):
    """
    implement to TorchSparse needed 
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        # Initialize separate buffers for clean data
        self.register_buffer('running_mean_clean', torch.zeros(num_features))
        self.register_buffer('running_var_clean', torch.ones(num_features))

        # Initialize separate buffers for corrupted data
        self.register_buffer('running_mean_corrupted', torch.zeros(num_features))
        self.register_buffer('running_var_corrupted', torch.ones(num_features))

        self.mode = 'mode_clean' ## mode_clean, mode_corrupted, mode_mixed

        self.cache_clean = None
        self.cache_corrupted = None
        self.cache_mixed = None

    def forward(self, input):
        self._check_input_dim(input)

        assert self.mode in ['mode_clean', 'mode_corrupted', 'mode_mixed'], "mode should be in 'mode_clean', 'mode_corrupted', 'mode_mixed'" 

        if self.mode == 'mode_clean':
            running_mean = self.running_mean_clean
            running_var = self.running_var_clean
            self.cache_clean = F.batch_norm(
                input, running_mean, running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
            return self.cache_clean
        elif self.mode == 'mode_corrupted':
            running_mean = self.running_mean_corrupted
            running_var = self.running_var_corrupted
            self.cache_corrupted = F.batch_norm(
                input, running_mean, running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
            return self.cache_corrupted
        else:
            running_mean = (self.running_mean_clean + self.running_mean_corrupted)*0.5
            std_clean = torch.sqrt(self.running_var_clean + self.eps)
            std_corrupted = torch.sqrt(self.running_var_corrupted + self.eps)
            running_std = (std_clean + std_corrupted)*0.5
            running_var = running_std ** 2
            self.cache_mixed = F.batch_norm(
                input, running_mean, running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
            return self.cache_mixed

class BatchNormMixer(BatchNorm1dMixer):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)