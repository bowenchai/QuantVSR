import torch
from spatial_correlation_sampler import SpatialCorrelationSampler

class Correlation(torch.nn.Module):
    """
    兼容 FlowNetC 原接口的包装器
    """
    def __init__(self, pad_size, kernel_size, max_displacement,
                 stride1, stride2, corr_multiply=1):
        super().__init__()
        # old → new 参数映射
        patch_size = (max_displacement * 2) // stride2 + 1
        self.corr = SpatialCorrelationSampler(
            kernel_size=kernel_size,
            patch_size=patch_size,
            stride=stride1,
            padding=0,
            dilation_patch=stride2
        )
        self.corr_multiply = corr_multiply

    def forward(self, x1, x2):
        out = self.corr(x1, x2)           # [B, H, W, Ph, Pw]
        if self.corr_multiply != 1:
            out = out * self.corr_multiply
        # 还原成 FlowNetC 期望的 [B, Ph*Pw, H, W]
        b, h, w, ph, pw = out.shape
        out = out.permute(0, 3, 4, 1, 2).reshape(b, ph * pw, h, w)
        return out
