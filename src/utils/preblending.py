import torch
import torch.nn.functional as F
from torchvision import transforms

def get_mask(mask, indexs=[1, 2,3,4,5,6,7,8,9,10,11,12,13, 14,15,16,17,18,19,20]):
    out = torch.zeros_like(mask, device=mask.device)
    for i in indexs:
        out[mask == i] = 1

    return out

def dilate_torch(x, dilate_kernel=17):
    return F.max_pool2d(
        x,
        kernel_size=dilate_kernel, stride=1, 
        padding=dilate_kernel//2
    )

def calc_pseudo_target_bg(target, target_parsing):
    gaussian_blur = transforms.GaussianBlur(3)
    por_mask = get_mask(target_parsing)
    por_mask = dilate_torch(por_mask, dilate_kernel=15)
    inp_mask = dilate_torch(por_mask, dilate_kernel=3) - por_mask
    inp_mask_idxes = inp_mask[0, 0].nonzero()
    por_mask_idxes = por_mask[0, 0].nonzero()
    idxes_dist = (((por_mask_idxes[:, None, :] - inp_mask_idxes[None, :, :]) / 512) ** 2).sum(dim=2)
    min_dist_idx = idxes_dist.min(dim=1)[1]
    inp_mask_colors = gaussian_blur(target)[0, :, inp_mask_idxes[:, 0], inp_mask_idxes[:, 1]].T
    min_dist_value = inp_mask_colors[min_dist_idx, :]
    pseudo_target = target.clone()
    pseudo_target[0, :, por_mask_idxes[:, 0], por_mask_idxes[:, 1]] = min_dist_value.T
    return pseudo_target
