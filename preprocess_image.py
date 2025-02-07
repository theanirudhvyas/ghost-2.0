import cv2
import numpy as np
from insightface.app import FaceAnalysis
from src.utils.crops import wide_crop_face, crop_face, emoca_crop
from src.utils.keypoint_detector import DECAKeypoints
import face_alignment


def preprocess_image(image, app, kpt_det):

    det = app.get(image)[0]
    face_wide = wide_crop_face(image, det['kps'])
    
    kpts = fa.get_landmarks_from_image(face_wide)
    
    if kpts is not None:
        kpts = np.vstack(kpts[0])
        if kpts.shape[0] == 68: #one face
            idx_68 = [0] #used to keep indices of good frames
            face_arc = crop_face(image, det['kps'])
            
            face_emoca = emoca_crop(face_wide[np.newaxis, ...], kpts[np.newaxis, ...])
            face_emoca = (face_emoca[0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            return {
                'face_arc': face_arc,
                'face_emoca': face_emoca,
                'face_wide': face_wide,
                'keypoints_68': kpts,
                'idx_68': idx_68
            }
        else:
            print('Multiple faces')
    else:
        print('No face detected')




if __name__ == '__main__':
    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda:0')

    image = cv2.imread('./examples/images/hab.jpg')

    res_dict = preprocess_image(image, app, fa)