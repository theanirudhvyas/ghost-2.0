import cv2
import numpy as np
from insightface.app import FaceAnalysis
from src.utils.crops import wide_crop_face, crop_face, emoca_crop
from src.utils.keypoint_detector import DECAKeypoints
# from PIL import Image

def preprocess_image(image, app, kpt_det):
    det = app.get(image)[0]
    face_arc = crop_face(image, det['kps'])
    face_wide = wide_crop_face(image, det['kps'])
    kpts, deca_crops = kpt_det(face_wide[np.newaxis, ...][ :, :, ::-1]) #[0, 224]
    face_emoca = emoca_crop(deca_crops, kpts)
    face_emoca = (face_emoca[0].permute(1, 2, 0).cpu().numpy()  * 255).astype(np.uint8)
    return {
        'face_arc': face_arc,
        'face_emoca': face_emoca,
        'face_wide': face_wide,
        'kpts': kpts.cpu().numpy()
    }





if __name__ == '__main__':
    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    deca = DECAKeypoints('cuda')

    image = cv2.imread('./examples/images/hab.jpg')

    res_dict = preprocess_image(image, app, deca)

    # Image.fromarray(res_dict['face_arc']).save('face_arc.png')
    # Image.fromarray(res_dict['face_emoca'].astype(np.uint8)).save('face_emoca.png')
    # Image.fromarray(res_dict['face_wide']).save('face_wide.png')
    