# GHOST 2.0: Generative High-fidelity One Shot Transfer of Heads

![teaser](https://github.com/user-attachments/assets/60b35f78-99d6-4c4a-90b4-b694402c9e6c)


We present GHOST 2.0, a novel approach for realistic head swapping. Our solution transfers head from source image to the target one in a natural way, seamessly blending with the target background and skin color. 

While the task of face swapping has recently gained attention in the research community, a related problem of head swapping remains largely unexplored. In addition to skin color transfer, head swap poses extra challenges, such as the need to preserve structural information of the whole head during synthesis and inpaint gaps between swapped head and background. In this paper, we address these concerns with GHOST 2.0, which consists of two problem-specific modules. First, we introduce a new Aligner model for head reenactment, which preserves identity information at multiple scales and is robust to extreme pose variations. Secondly, we use a Blender module that seamlessly integrates the reenacted head into the target background by transferring skin color and inpainting mismatched regions. Both modules outperform the baselines on the corresponding tasks, allowing to achieve state-of-the-art results in head swapping. We also tackle complex cases, such as large difference in hair styles of source and driver. For more details on our solution, please refer to the paper.

## Installation
Install conda environment:
```
conda create -n ghost python=3.10

pip install face-alignment
pip install facenet_pytorch

conda config --add channels conda-forge
conda config --set channel_priority strict
pip install -U 'git+https://github.com/facebookresearch/fvcore'
conda install pytorch3d -c pytorch3d-nightly

pip install -r requirements.txt
python -m pip uninstall numpy
python -m pip install numpy==1.23.1
```
Clone the following repositories in the ```repos``` folder. Download respective checkpoints:

[DECA](https://github.com/yfeng95/DECA)

[EMOCA](https://github.com/anastasia-yaschenko/EMOCA)

[BlazeFace](https://github.com/anastasia-yaschenko/BlazeFace_PyTorch)

[stylematte](https://github.com/chroneus/stylematte)

For EMOCA, download ResNet50 folder from [here](https://github.com/anastasia-yaschenko/emoca/releases/tag/resnet) and unpack into ```repos/emoca/gdl_apps/EmotionRecognition/``` and ```assets/EmotionRecognition/image_based_networks/```

Download the models from releases. Place them into the following folders
```
- aligner_checkpoints
    - aligner_1020_gaze_final.ckpt
- blender_checkpoints
    - blender_lama.ckpt

- src
    - losses
        - gaze_models
 
- weights
    - backbone50_1.pth
    - vgg19-d01eb7cb.pth
    - segformer_B5_ce.onnx
```

## Inference
For inference run
```
python inference.py --source ./examples/images/hab.jpg --target ./examples/images/elon.jpg --save_path result.png
```

## Training
1. Download VoxCeleb2. We expect the following structure of data directory and assume images are stored in BGR format:
```
- train
    - person_id
        - video_folder
            -h5 video files
-test
```
2. Preprocess data according to the example in file ```preprocess_image.py```
3. To train Aligner, run ```python train_aligner.py```
4. To train Blender, run ```python train_blender.py```

## Citation

## Acknowledgements
This work utilizes code from the follwing repositories:

[Neural Head Reenactment with Latent Pose Descriptors](https://github.com/shrubb/latent-pose-reenactment)

[EMOPortraits: Emotion-enhanced Multimodal One-shot Head Avatars](https://github.com/neeek2303/EMOPortraits)

[RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments](https://github.com/Tobias-Fischer/rt_gene?tab=readme-ov-file)
