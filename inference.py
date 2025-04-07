import cv2
import torch
import argparse
import yaml
import time
import datetime
from torchvision import transforms
import onnxruntime as ort
from PIL import Image
import importlib
import sys
from omegaconf import OmegaConf
from torchvision.transforms.functional import rgb_to_grayscale
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Timing decorator for performance monitoring
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"[TIMING] {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

# Add compatibility layer for NMS operator
def patch_insightface_nms():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        return FaceAnalysis
    except RuntimeError as e:
        if 'torchvision::nms' in str(e):
            # Monkey patch the NMS function in insightface
            import types
            from torchvision.ops import nms
            import torch
            
            # Create a custom NMS function that matches the expected interface
            def custom_nms(boxes, scores, iou_threshold):
                # Convert to torch tensors if they're not already
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores, dtype=torch.float32)
                
                # Call torchvision's NMS
                keep = nms(boxes, scores, iou_threshold)
                return keep.cpu().numpy() if keep.is_cuda else keep.numpy()
            
            # Add the custom NMS function to torchvision.ops
            if not hasattr(torch.ops, 'torchvision'):
                torch.ops.torchvision = types.SimpleNamespace()
            torch.ops.torchvision.nms = custom_nms
            
            # Try importing again
            from insightface.app import FaceAnalysis
            return FaceAnalysis
        else:
            raise e

# Get patched FaceAnalysis class
FaceAnalysis = patch_insightface_nms()

from src.utils.crops import *
from repos.stylematte.stylematte.models import StyleMatte
from src.utils.inference import *
from src.utils.inpainter import LamaInpainter
from src.utils.preblending import calc_pseudo_target_bg, post_inpainting
from train_aligner import AlignerModule
from train_blender import BlenderModule

def main(args):
    print("\n[INFO] Starting GHOST 2.0 inference...")
    print(f"[INFO] Source image: {args.source}")
    print(f"[INFO] Target image: {args.target}")
    print(f"[INFO] Output will be saved to: {args.save_path}")

    print("\n[INFO] Loading configuration files...")
    with open(args.config_a, "r") as stream:
        cfg_a = OmegaConf.load(stream)

    with open(args.config_b, "r") as stream:
        cfg_b = OmegaConf.load(stream)
    print("[INFO] Configuration files loaded successfully")

    print("\n[INFO] Loading aligner model...")
    aligner = AlignerModule(cfg_a)
    ckpt = torch.load(args.ckpt_a, map_location='cpu')
    aligner.load_state_dict(torch.load(args.ckpt_a), strict=False)
    aligner.eval()
    aligner.cuda()
    print("[INFO] Aligner model loaded successfully")

    print("\n[INFO] Loading blender model...")
    blender = BlenderModule(cfg_b)
    blender.load_state_dict(torch.load(args.ckpt_b, map_location='cpu')["state_dict"], strict=False,)
    blender.eval()
    blender.cuda()
    print("[INFO] Blender model loaded successfully")

    print("\n[INFO] Initializing LamaInpainter...")
    inpainter = LamaInpainter()
    print("[INFO] LamaInpainter initialized successfully")

    print("\n[INFO] Initializing face detection model...")
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] Face detection model initialized with CUDA")
    except Exception as e:
        print(f"[WARNING] Error initializing FaceAnalysis with CUDA: {e}")
        print("[INFO] Falling back to CPU execution...")
        app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] Face detection model initialized with CPU")

    if args.use_kandi:
        print("\n[INFO] Loading Kandinsky inpainting model...")
        pipe = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        print("[INFO] Kandinsky inpainting model loaded successfully")

    print("\n[INFO] Loading segmentation model (StyleMatte)...")
    segment_model = StyleMatte()
    segment_model.load_state_dict(
        torch.load(
            './repos/stylematte/stylematte/checkpoints/stylematte_synth.pth',
            map_location='cpu'
        )
    )
    segment_model = segment_model.cuda()
    segment_model.eval()
    print("[INFO] Segmentation model loaded successfully")

    print("\n[INFO] Loading ONNX parsing model...")
    providers = [
       ("CUDAExecutionProvider", {})
    ]
    parsings_session = ort.InferenceSession('./weights/segformer_B5_ce.onnx', providers=providers)
    input_name = parsings_session.get_inputs()[0].name
    output_names = [output.name for output in parsings_session.get_outputs()]
    print("[INFO] ONNX parsing model loaded successfully")
    
    mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
    std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]
    
    infer_parsing = lambda img: torch.tensor(
        parsings_session.run(output_names, {
            input_name: (((img[:, [2, 1, 0], ...] / 2 + 0.5).cpu().detach().numpy() - mean) / std).astype(np.float32)
        })[0],
        device='cuda',
        dtype=torch.float32
    )

    @timing_decorator
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

    @timing_decorator
    def process_img(img_path, target=False):
        full_frames = cv2.imread(img_path)
        dets = app.get(full_frames)
        kps = dets[0]['kps']
        wide = wide_crop_face(full_frames, kps, return_M=target)
        if target:
            wide, M = wide
        arc = norm_crop(full_frames, kps)
        mask = calc_mask(wide)
        arc = normalize_and_torch(arc)
        wide = normalize_and_torch(wide)
        if target:
            return wide, arc, mask, full_frames, M
        return wide, arc, mask

    print("\n[INFO] Processing source image...")
    wide_source, arc_source, mask_source = process_img(args.source)
    print("[INFO] Source image processed successfully")
    
    print("\n[INFO] Processing target image...")
    wide_target, arc_target, mask_target, full_frames, M = process_img(args.target, target=True)
    print("[INFO] Target image processed successfully")
    

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
    
    print("\n[INFO] Running aligner model...")
    with torch.no_grad():
        output = aligner(X_dict)
    print("[INFO] Aligner model completed successfully")


    target_parsing = infer_parsing(wide_target)
    pseudo_norm_target = calc_pseudo_target_bg(wide_target, target_parsing)
    soft_mask = calc_mask(((output['fake_rgbs'] * output['fake_segm'])[0, [2, 1, 0], :, :] + 1) / 2)[None]
    new_source = output['fake_rgbs'] * soft_mask[:, None, ...] + pseudo_norm_target * (1 - soft_mask[:, None, ...])

    blender_input = {
        'face_source': new_source, # output['fake_rgbs']*output['fake_segm'] + norm_target*(1-output['fake_segm']),# face_source,
        'gray_source': rgb_to_grayscale(new_source[0][[2, 1, 0], ...]).unsqueeze(0),
        'face_target': wide_target,
        'mask_source': infer_parsing(output['fake_rgbs']*output['fake_segm']),
        'mask_target': target_parsing,
        'mask_source_noise': None,
        'mask_target_noise': None,
        'alpha_source': soft_mask
    }

    print("\n[INFO] Running blender model...")
    output_b = blender(blender_input, inpainter=inpainter)
    print("[INFO] Blender model completed successfully")

    print("\n[INFO] Generating final output...")
    np_output = np.uint8((output_b['oup'][0].detach().cpu().numpy().transpose((1, 2, 0))[:,:,::-1] / 2 + 0.5)*255)
    result = copy_head_back(np_output, full_frames[..., ::-1], M)
    if args.use_kandi:
        print("[INFO] Applying post-inpainting with Kandinsky...")
        result = post_inpainting(result, output, full_frames, M, infer_parsing, pipe)
        print("[INFO] Post-inpainting completed successfully")
    
    print(f"\n[INFO] Saving result to {args.save_path}...")
    Image.fromarray(result).save(args.save_path)
    print(f"[INFO] Result saved successfully to {args.save_path}")
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    minutes = int(overall_duration // 60)
    seconds = int(overall_duration % 60)
    print(f"\n[INFO] GHOST 2.0 inference completed successfully!")
    print(f"[INFO] Total execution time: {minutes} minutes and {seconds} seconds")
    print(f"[INFO] Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    

if __name__ == "__main__":
    print("\n===== GHOST 2.0 Inference =====\n")
    print(f"[INFO] Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    overall_start_time = time.time()
    parser = argparse.ArgumentParser()
    
    # Generator params
    parser.add_argument('--config_a', default='./configs/aligner.yaml', type=str, help='Path to Aligner config')
    parser.add_argument('--config_b', default='./configs/blender.yaml', type=str, help='Path to Blender config')
    parser.add_argument('--source', default='./examples/images/hab.jpg', type=str, help='Path to source image')
    parser.add_argument('--target', default='./examples/images/elon.jpg', type=str, help='Path to target image')
    parser.add_argument('--use_kandi',action='store_true', help='Usage post-blending step')
    parser.add_argument('--ckpt_a', default='./aligner_checkpoints/aligner_1020_gaze_final.ckpt', type=str, help='Aligner checkpoint')
    parser.add_argument('--ckpt_b', default='./blender_checkpoints/blender_lama.ckpt', type=str, help='Blender checkpoint')
    parser.add_argument('--save_path', default='result.png', type=str, help='Path to save the result')
    
    args = parser.parse_args()
    main(args)
