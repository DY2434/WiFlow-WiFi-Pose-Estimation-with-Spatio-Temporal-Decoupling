import torch

def time_masking(x, mask_ratio=0.3, mask_len_range=(5, 10)):
    """时间掩码增强"""
    device = x.device
    masked_x = x.clone()
    B, C, T = masked_x.shape

    for i in range(B):
        if torch.rand(1).item() < mask_ratio:
            num_masks = torch.randint(1, 3, (1,)).item()
            for _ in range(num_masks):
                mask_len = torch.randint(mask_len_range[0], mask_len_range[1], (1,)).item()
                start = torch.randint(0, T - mask_len, (1,)).item()
                for c in range(C):
                    mean_val = masked_x[i, c, :].mean()
                    masked_x[i, c, start:start + mask_len] = mean_val

    return masked_x


def add_noise(x, noise_level=0.05):
    """添加高斯噪声"""
    device = x.device
    noise = torch.randn_like(x).to(device) * noise_level * torch.std(x)
    return x + noise


def random_scaling(x, scale_range=(0.9, 1.1)):
    """随机缩放信号幅度"""
    device = x.device
    if torch.rand(1).item() < 0.5:
        scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).to(device)
        return x * scale_factor
    return x