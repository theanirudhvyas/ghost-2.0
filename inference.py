import cv2
import torch
import argparse
import yaml
from torchvision import transforms
from PIL import Image
from insightface.app import FaceAnalysis
from src.utils.crops import *
from repos.stylematte.stylematte.models import StyleMatte
from src.utils.inference import *
from train_aligner import AlignerModule

def main(args):

    with open(args.config_a, "r") as stream:
        cfg = yaml.safe_load(stream)

    aligner = AlignerModule(cfg)
    ckpt = torch.load(args.ckpt_a, map_location='cpu')
    aligner.load_state_dict(torch.load(args.ckpt_a)["state_dict"], strict=False)
    aligner.eval()
    aligner.cuda()

    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    segment_model = StyleMatte()
    segment_model.load_state_dict(
        torch.load(
            './repos/stylematte/stylematte/checkpoints/drive-download-20230511T084109Z-001/stylematte_synth.pth',
            map_location='cpu'
        )
    )
    segment_model = segment_model.cuda()
    segment_model.eval()

    def calc_mask(img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).cuda()
        if img.max() > 1.:
            img = img / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input_t = normalize(img)
        input_t = input_t.unsqueeze(0).float()
        with torch.no_grad():
            out = segment_model(input_t)
        result = out[0]
    
        return result[0]

    def process_img(img_path):
        full_frames = cv2.imread(img_path)
        dets = app.get(full_frames)
        kps = dets[0]['kps']
        wide = wide_crop_face(full_frames, kps)
        arc = norm_crop(full_frames, kps)
        mask = calc_mask(wide)
        arc = normalize_and_torch(arc)
        wide = normalize_and_torch(wide)
        return wide, arc, mask

    wide_source, arc_source, mask_source = process_img(args.source)
    wide_target, arc_target, mask_target = process_img(args.target)

    wide_source = wide_source.unsqueeze(1)
    arc_source = arc_source.unsqueeze(1)
    source_mask = mask_source.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    target_mask = mask_target.unsqueeze(0).unsqueeze(0)

    X_dict = {
        'source': {
            'face_arc': arc_source,
            'face_wide': wide_source * mask_source,
            'face_wide_mask': mask_source
        },
        'target': {
            'face_arc': arc_target,
            'face_wide': wide_target * mask_target,
            'face_wide_mask': mask_target
        }
    }
    
    with torch.no_grad():
        output = aligner(X_dict)

    mask_fake = calc_mask(output['fake_rgbs'][0] / 2 + 0.5)
    fake = output['fake_rgbs'][0] * mask_fake
    np_output = np.uint8((fake.cpu().numpy().transpose((1, 2, 0))[:,:,::-1] / 2 + 0.5)*255)
    Image.fromarray(np_output).save(args.save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Generator params
    parser.add_argument('--config_a', default='./configs/aligner.yaml', type=str, help='Path to Aligner config')
    parser.add_argument('--source', default='./examples/images/hab.jpg', type=str, help='Path to source image')
    parser.add_argument('--target', default='./examples/images/elon.jpg', type=str, help='Path to target image')
    parser.add_argument('--ckpt_a', default='/home/jovyan/yaschenko/headswap/HeSerAligner_keypoints/final_models/dis_blocks_6_512_adv_w_0.1_rtgene_w1_ep1000/checkpoints/epoch-1020.ckpt', type=str, help='Aligner checkpoint')
    parser.add_argument('--save_path', default='result.png', type=str, help='Path to save the result')
    
    args = parser.parse_args()
    main(args)