import os
os.environ['HDF5_PLUGIN_PATH'] = '/home/jovyan/paramonov/HeadSwapNoRuns/h5_jpeg'
import sys
import torch
from glob import glob
from collections import defaultdict
import h5py
import numpy as np
import random
from typing import Optional
import torchvision
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms.functional as TF
import torchvision
from repos.emoca.gdl.datasets.ImageDatasetHelpers import bbox2point
import pickle
from torchvision.utils import make_grid

def dict_factory():
    return defaultdict(dict)

class Voxceleb2H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path: str,
        source_len: int,
        samples_cnt: Optional[int] = None,
        shuffle=False,
        transform=None,
        subset_size=None,
        flip_transform=False,
        return_masks=True, 
        image_size=512,
        cross=False
    ) -> None:
        self.root_path = root_path
        self.source_len = source_len
        self.image_size = image_size
        self.cross = cross
        self.return_masks = return_masks
        self.flip_transform = flip_transform
        self.h5_dict_tree = defaultdict(dict_factory)
        self.valid_idxs = []
        self.samples_cnt = samples_cnt
        
        with open('/home/jovyan/yaschenko/dev_masks_parsings/iqa_final_filtered_clip_0.2_hyper_60_sharp_0.3.pkl','rb') as f:
            h5_iqa = pickle.load(f)
    
        self.h5_paths = sorted(list(h5_iqa.keys()))[:50]
        
        if shuffle:
            rng = random.Random(46)
            rng.shuffle(self.h5_paths)

        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if transform is None:
            transform = self.to_tensor
        self.transform = transform
        
        
        for idx, h5_path in enumerate(self.h5_paths):
            if subset_size is not None:
                if idx >= subset_size:
                    break
            head, h5_file = os.path.split(h5_path)
            head, ref = os.path.split(head)
            head, id = os.path.split(head)
            
            try:
                with h5py.File(h5_path) as f:
                    seg_len = len(f['face_wide_mask'])
                    if self.return_masks:
                        assert 'face_wide_mask' in f
                    assert 'keypoints_68' in f
                    
            except Exception as e:
                print(h5_path, e)
                continue
            
            if seg_len < (self.source_len + 1):
                continue
            
            self.h5_dict_tree[id][ref][h5_file] = {'idx': idx, 'seg_len': seg_len}
            self.valid_idxs.append(idx)
    
    def get_sequence(self, video_tensor, seed, is_flip=False):
        result = torch.stack([
            self.transform(img) if i == (video_tensor.shape[0] - 1) else self.to_tensor(img)
            for i, img in enumerate(video_tensor[:])
        ])
        return result
        

    def flip(self, x, is_image=True):
        if is_image:
            x[-1] = torchvision.transforms.functional.hflip(x[-1])
        else:
            x[-1][..., 0] = 512 - x[-1][..., 0] #flip keypoints
        return x
            
    
    def __getitem__(self, idx):
        
        h5_path = self.h5_paths[idx]
        seed = torch.randint(0, 0x7fff_ffff_ffff_ffff, (1,))
        
        with h5py.File(h5_path) as f:
            seg_len = len(f['face_wide_mask'])
            idx_mask = f['idx_68'] #indices of frames with detected keypoints

            idxs = idx_mask[np.sort(np.random.choice(len(idx_mask), size=self.source_len if self.cross else (self.source_len + 1), replace=False)).tolist()].tolist()

            if self.cross:
                permutation = [0]
            else:
                target_idx = np.random.choice(len(idxs))
                permutation = list(range(len(idxs)))
                permutation.pop(target_idx)
                permutation.append(target_idx)
            
            face_arc = f['face_arc'][idxs][permutation]
            face_wide = f['face_wide'][idxs][permutation]
        
            if self.return_masks:
                face_wide_mask = torch.stack([TF.to_tensor(x) for x in f['face_wide_mask'][idxs][permutation]]) 
                
            segmentation = torch.stack([TF.to_tensor(x) for x in f['face_wide_parsing_segformer_B5_ce'][idxs][permutation]])
            
            try:
                face_keypoints = f['keypoints_68'][idxs][permutation]
            except Exception as e:
                print(e)
                print(h5_path)
                face_keypoints = torch.zeros((seg_len, 68, 2), dtype=torch.int16)

        #get target additionally for cross reenactment
        if self.cross:
            idx_target = np.random.randint(0, self.__len__())
            h5_path_target = self.h5_paths[idx_target]
            
            with h5py.File(h5_path_target) as f_target:
                seg_len_target = len(f_target['face_wide_mask'])
                idx_mask_target = f_target['idx_68'] #indices of frames with detected keypoints
    
                idxs_target = idx_mask_target[np.random.choice(len(idx_mask_target), size=1, replace=False).tolist()].tolist()
                
                face_arc = np.vstack([face_arc, f_target['face_arc'][idxs_target]])
                face_wide = np.vstack([face_wide, f_target['face_wide'][idxs_target]])
            
                if self.return_masks:
                    
                    face_wide_mask_target = TF.to_tensor(f_target['face_wide_mask'][idxs_target][0]).unsqueeze(0)
                    face_wide_mask = torch.cat([face_wide_mask, face_wide_mask_target], 0)
                    
                segmentation_target =  TF.to_tensor(f_target['face_wide_parsing_segformer_B5_ce'][idxs_target][0]).unsqueeze(0)
                segmentation = torch.cat([segmentation, segmentation_target], 0)
                
                try:
                    face_keypoints_target = f_target['keypoints_68'][idxs_target]
                except Exception as e:
                    print(e)
                    print(h5_path_target)
                    face_keypoints_target = torch.zeros((seg_len_target, 68, 2), dtype=torch.int16)
                face_keypoints = np.stack([face_keypoints, face_keypoints_target])
        
        
        face_arc = self.get_sequence(face_arc, seed)
        face_wide = self.get_sequence(face_wide, seed)

        crop = self.crop_face(face_keypoints, face_wide, face_wide_mask)
                    
        # if self.flip_transform:
        p = torch.rand(1)
        if p < 0.5:
            face_arc = self.flip(face_arc)
            face_wide = self.flip(face_wide)
            face_wide_mask = self.flip(face_wide_mask)
            segmentation = self.flip(segmentation)
            crop = self.flip(crop)
            face_keypoints = self.flip(face_keypoints, is_image=False)
            
        


        if self.image_size != 512:
            face_wide = torch.nn.functional.interpolate(face_wide, size=(self.image_size, self.image_size), mode='bilinear')
            face_wide_mask = torch.nn.functional.interpolate(face_wide_mask, size=(self.image_size, self.image_size), mode='bilinear')
            segmentation = torch.nn.functional.interpolate(segmentation, size=(self.image_size, self.image_size), mode='bilinear')
            # face_keypoints = face_keypoints * (self.image_size // 512)

        
        result = {
            'face_arc': face_arc,
            'face_wide': face_wide
        }
        if self.return_masks:
            result['face_wide_mask'] = face_wide_mask

        result['crop_emotion'] = crop[0].to(torch.float16)#.unsqueeze(0)
        result['keypoints_target'] = torch.tensor(face_keypoints[-1]).to(torch.int16)
        result['segmentation'] = segmentation
        return result

    def crop_face(self, face_keypoints, face_wide, face_wide_mask):
        try:
            scale = 1.25
            left = np.min(face_keypoints[-1, :, 0])
            right = np.max(face_keypoints[-1, :, 0])
            top = np.min(face_keypoints[-1, :, 1])
            bottom = np.max(face_keypoints[-1, :, 1])
            old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            size = int(old_size * scale)
            half = size // 2
            new_left = int(center[0] - half)
            new_right = int(center[0] + half)
            new_top = int(center[1] - half)
            new_bottom = int(center[1] + half)
            
            crop = TF.crop(face_wide[-1] * face_wide_mask[-1], new_top, new_left, (new_bottom - new_top), (new_right - new_left))
            crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=((224, 224)), mode='bilinear')
        except Exception as e:
            print(e)
            crop = torch.nn.functional.interpolate((face_wide[-1] * face_wide_mask[-1]).unsqueeze(0), size=((224, 224)), mode='bilinear')
            
        return crop
        

    def __len__(self):
        dataset_len = len(self.valid_idxs)
        if self.samples_cnt is not None:
            dataset_len = min(dataset_len, self.samples_cnt)
        return dataset_len