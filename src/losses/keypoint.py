import torch.nn as nn
from torchvision.transforms.functional import to_tensor, to_pil_image
import repos.DECA.decalib.utils.config as config
from repos.DECA.decalib.deca import DECA
from repos.DECA.decalib.utils import util
import torch
import torchvision
import os
from repos.BlazeFace_PyTorch.blazeface import BlazeFace

class KeypointLoss(nn.Module):
    def __init__(self, mode='all', device='cuda'):
        super().__init__()
        device='cuda'
        cfg = config.cfg
        config.cfg.deca_dir = '/home/jovyan/yaschenko/headswap/repos/DECA/'
        
        cfg.model.topology_path = os.path.join(config.cfg.deca_dir, 'data', 'head_template.obj')
        cfg.model.addfiles_path = 'data'
        cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl')
        cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy')
        cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png')
        cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png')
        cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
        cfg.model.use_tex = False
        # cfg.dataset.image_size = 224
        # cfg.model.uv_size = 224
        cfg.rasterizer_type = 'pytorch3d'
        cfg.model.extract_tex = False
    
        self.deca = DECA(cfg, device)
        self.deca.eval()

        self.front_net = BlazeFace().to(device)
        self.front_net.load_weights("/home/jovyan/yaschenko/headswap/repos/BlazeFace_PyTorch/blazeface.pth")
        self.front_net.load_anchors("/home/jovyan/yaschenko/headswap/repos/BlazeFace_PyTorch/anchors.npy")
        self.front_net.eval()

        self.mode = mode
        all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        self.register_buffer('contour', torch.LongTensor(all), persistent=False)

        self.register_buffer('weights', torch.ones(68), persistent=False)
        self.weights[5:7] = 2.0
        self.weights[10:12] = 2.0
        self.weights[27:36] = 1.5
        self.weights[30] = 3.0
        self.weights[31] = 3.0
        self.weights[35] = 3.0
        self.weights[60:68] = 1.5
        self.weights[48:60] = 1.5
        self.weights[48] = 3
        self.weights[54] = 3
        
        if self.mode == 'expr':
            self.register_buffer('upper_lids', torch.LongTensor([37, 38, 43, 44]), persistent=False)
            self.register_buffer('lower_lids', torch.LongTensor([41, 40, 47, 46]), persistent=False)
            self.register_buffer('upper_lips', torch.LongTensor([61, 62, 63]), persistent=False)
            self.register_buffer('lower_lips', torch.LongTensor([67, 66, 65]), persistent=False)

    def forward(self, pred, gt):

        with torch.no_grad():
            img_batch_128 = torch.nn.functional.interpolate((gt[:, [2, 1, 0], :, :] / 2 + 0.5) * 255, size=(128, 128))
            front_detections = self.front_net.predict_on_batch(img_batch_128)
            front_detections = torch.cat(front_detections, 0)
            front_detections = torch.stack([front_detections[:, 0] - 0.1, front_detections[:, 1] - 0.1, front_detections[:, 2] + 0.1, front_detections[:, 3] + 0.1], 1) 
    
            b, c, h, w = gt.shape
    
            top = (front_detections[:, 0] * h).int()
            left = (front_detections[:, 1]  * w).int()
            height = ((front_detections[:, 2] - front_detections[:, 0]) * h).int()
            width = ((front_detections[:, 3] - front_detections[:, 1]) * w).int()

        cropped_preds = []
        cropped_gt = []
        
        for i in range(pred.size(0)):
            try:
                img_cropped_pred = torchvision.transforms.functional.crop(pred[i], top[i], left[i], height[i], width[i])
                img_cropped_gt = torchvision.transforms.functional.crop(gt[i], top[i], left[i], height[i], width[i])
            except IndexError:
                img_cropped_pred = pred[i]
                img_cropped_gt = gt[i]
                
            img_cropped_pred = torch.nn.functional.interpolate(img_cropped_pred[[2, 1, 0], :, :].unsqueeze(0) / 2 + 0.5, size=(224, 224))
            img_cropped_gt = torch.nn.functional.interpolate(img_cropped_gt[[2, 1, 0], :, :].unsqueeze(0) / 2 + 0.5, size=(224, 224))
                
            cropped_preds.append(img_cropped_pred)
            cropped_gt.append(img_cropped_gt)
            
        cropped_preds = torch.cat(cropped_preds)

        cropped_gt = torch.cat(cropped_gt)
        
        
        codedict_pred = self.deca.encode(cropped_preds)
        _, landmarks2d_pred, _ = self.deca.flame(shape_params=codedict_pred['shape'], expression_params=codedict_pred['exp'], pose_params=codedict_pred['pose'])
        landmarks2d_pred = util.batch_orth_proj(landmarks2d_pred, codedict_pred['cam'])[:,:,:2]; landmarks2d_pred[:,:,1:] = -landmarks2d_pred[:,:,1:]

        with torch.no_grad():
            codedict_gt = self.deca.encode(cropped_gt)
        _, landmarks2d_gt, _ = self.deca.flame(shape_params=codedict_gt['shape'], expression_params=codedict_gt['exp'], pose_params=codedict_gt['pose'])
        landmarks2d_gt = util.batch_orth_proj(landmarks2d_gt, codedict_gt['cam'])[:,:,:2]; landmarks2d_gt[:,:,1:] = -landmarks2d_gt[:,:,1:]
        
        if self.mode == 'all':
            diff = landmarks2d_pred - landmarks2d_gt
            loss = (diff.abs().mean(-1) * self.weights[None] / self.weights.sum()).sum(-1).mean()
            return loss

        if self.mode == 'expr':
            diff_pred = landmarks2d_pred[:, self.upper_lips] - landmarks2d_pred[:, self.lower_lips]
            diff = landmarks2d_gt[:, self.upper_lips] - landmarks2d_gt[:, self.lower_lips]
            loss_lips = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

            diff_pred = landmarks2d_pred[:, self.upper_lids] - landmarks2d_pred[:, self.lower_lids]
            diff = landmarks2d_gt[:, self.upper_lids] - landmarks2d_gt[:, self.lower_lids]
    
            loss_eyes = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()
            return loss_eyes + loss_lips