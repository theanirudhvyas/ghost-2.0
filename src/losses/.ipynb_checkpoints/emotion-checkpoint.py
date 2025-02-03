import torch.nn as nn
from pathlib import Path
import os
os.sys.path.append('/home/jovyan/yaschenko/headswap/repos/emoca')
from repos.emoca.gdl_apps.EmotionRecognition.utils.io import load_model, test
from repos.emoca.gdl.datasets.ImageDatasetHelpers import bbox2point
import numpy as np
import torchvision
import torch
from torchvision.transforms.functional import to_pil_image


class EmotionLoss(nn.Module):

    def __init__(self, path_to_models="/home/jovyan/yaschenko/headswap/repos/emoca/assets/EmotionRecognition/image_based_networks", model_name = "ResNet50"):
        super().__init__()
        self.model = load_model(Path(path_to_models) / model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, pred, target, kpts):
    
        scale = 1.25
        kpts = kpts.cpu().numpy()
        left = np.min(kpts[:, :, 0], axis=1)
        right = np.max(kpts[:, :, 0], axis=1)
        top = np.min(kpts[:, :, 1], axis=1)
        bottom = np.max(kpts[:, :, 1], axis=1)
        old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        
        size = (old_size * scale).astype(np.int16)
        new_left = (center[:, 0] - size // 2).astype(np.int16)
        new_right = (center[:, 0] + size // 2).astype(np.int16)
        new_top = (center[:, 1] - size // 2).astype(np.int16)
        new_bottom = (center[:, 1] + size // 2).astype(np.int16)

        crop_list = []
        for i in range(pred.size(0)):
            try:
                crop = torchvision.transforms.functional.crop(pred[i], new_top[i], new_left[i], (new_bottom[i] - new_top[i]), (new_right[i] - new_left[i]))
                crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=((224, 224)), mode='bilinear')
            except:
                crop = torch.nn.functional.interpolate(pred[i].unsqueeze(0), size=((224, 224)), mode='bilinear')
            crop_list.append(crop
                            )
        crop_list = torch.cat(crop_list, dim=0)
        batch = {'image': crop_list[:, [2, 1, 0], :, :] / 2 + 0.5}
        output_pred = self.model(batch)['features']
        
        with torch.no_grad():
            batch = {'image': target[:, [2, 1, 0], :, :] / 2 + 0.5}
            output_target = self.model(batch)['features']

        loss = 0
        for f_pred, f_target in zip(output_pred, output_target):
            loss += torch.nn.functional.l1_loss(f_pred, f_target)

        return loss
        
        