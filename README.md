# GHOST 2.0: Generative High-fidelity One Shot Transfer of Heads

<a href='https://arxiv.org/abs/2502.18417'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>        <a href='https://huggingface.co/spaces/ai-forever/GHOST-2.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>        <a href='https://ai-forever.github.io/ghost-2.0/'><img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue'></a>

![teaser](https://github.com/user-attachments/assets/60b35f78-99d6-4c4a-90b4-b694402c9e6c)


We present GHOST 2.0, a novel approach for realistic head swapping. Our solution transfers head from source image to the target one in a natural way, seamessly blending with the target background and skin color.

While the task of face swapping (see our [GHOST](https://github.com/ai-forever/ghost) model) has recently gained attention in the research community, a related problem of head swapping remains largely unexplored. In addition to skin color transfer, head swap poses extra challenges, such as the need to preserve structural information of the whole head during synthesis and inpaint gaps between swapped head and background. We address these concerns with GHOST 2.0, which consists of two problem-specific modules. First, we introduce a new Aligner model for head reenactment, which preserves identity information at multiple scales and is robust to extreme pose variations. Secondly, we use a Blender module that seamlessly integrates the reenacted head into the target background by transferring skin color and inpainting mismatched regions. Both modules outperform the baselines on the corresponding tasks, allowing to achieve state-of-the-art results in head swapping. We also tackle complex cases, such as large difference in hair styles of source and driver. For more details on our solution, please refer to the paper.

## Installation

### Environment Setup

1. Create a new conda environment with Python 3.10:
```bash
conda create -n ghost-new python=3.10
conda activate ghost-new
```

2. Install PyTorch and other core dependencies:
```bash
# Install PyTorch with CUDA support (if available)
pip install torch==2.2.2 torchvision==0.17.2

# Install face detection and alignment libraries
pip install face-alignment>=1.3.5
pip install facenet_pytorch>=2.5.2
pip install insightface==0.7.3
```

3. Install 3D modeling dependencies:
```bash
# Add conda-forge channel for pytorch3d
conda config --add channels conda-forge
conda config --set channel_priority strict
pip install -U 'git+https://github.com/facebookresearch/fvcore'
conda install pytorch3d -c pytorch3d-nightly
```

4. Install all other required packages:
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Important: Some packages require numpy<2.0.0
python -m pip install numpy==1.26.4
```

### Important Notes
- If you encounter NumPy version conflicts, make sure to use a version below 2.0.0 as some packages are not compatible with NumPy 2.x
- If you encounter CUDA compatibility issues, try using CPU execution by modifying the providers in the code
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

### Running the Model
For inference, you can either use the Jupyter notebook ```inference.ipynb``` or run the Python script directly:

```bash
# Using the provided script
./run_inference.sh

# Or run directly with Python
python inference.py --source ./examples/images/hab.jpg --target ./examples/images/elon.jpg --save_path result.png
```

### Troubleshooting
If you encounter any dependency issues during inference:

1. Check that all required packages are installed:
```bash
pip install -r requirements.txt
```

2. Some specific packages that might need manual installation:
```bash
pip install transformers simple-lama-inpainting lightning
```

3. If you encounter CUDA errors, the code will automatically fall back to CPU execution

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
If you find our method useful in your research, please consider citing
```
@misc{groshev2025ghost2
      title={GHOST 2.0: generative high-fidelity one shot transfer of heads},
      author={Alexander Groshev and Anastasiia Iashchenko and Pavel Paramonov and Denis Dimitrov and Andrey Kuznetsov},
      year={2025},
      eprint={2502.18417},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.18417},
}
```

## Ethics
Deep fake synthesis methods have been improved a lot in quality in recent years. The research solutions were wrapped in easy-to-use API, software and different plugins for people with a little technical knowledge. As a result, almost anyone is able to make a deepfake image or video by just doing a short list of simple operations. At the same time, a lot of people with malicious intent are able to use this technology in order to produce harmful content. High distribution of such a content over the web leads to caution, disfavor and other negative feedback to deepfake synthesis or head swap research.

As a group of researchers, we are not trying to denigrate celebrities and statesmen or to demean anyone. We are computer vision researchers, we are engineers, we are activists, we are hobbyists, we are human beings. To this end, we feel that it's time to come out with a standard statement of what this technology is and isn't as far as us researchers are concerned.

GHOST 2.0 is not for creating inappropriate content.
GHOST 2.0 is not for changing heads without consent or with the intent of hiding its use.
GHOST 2.0 is not for any illicit, unethical, or questionable purposes.
GHOST 2.0 exists to experiment and discover AI techniques, for social or political commentary, for movies, and for any number of ethical and reasonable uses.
We are very troubled by the fact that GHOST 2.0 can be used for unethical and disreputable things. However, we support the development of tools and techniques that can be used ethically as well as provide education and experience in AI for anyone who wants to learn it hands-on. Now and further, we take a zero-tolerance approach and total disregard to anyone using this software for any unethical purposes and will actively discourage any such uses.

## Acknowledgements
This work utilizes code from the follwing repositories:

[Neural Head Reenactment with Latent Pose Descriptors](https://github.com/shrubb/latent-pose-reenactment)

[EMOPortraits: Emotion-enhanced Multimodal One-shot Head Avatars](https://github.com/neeek2303/EMOPortraits)

[RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments](https://github.com/Tobias-Fischer/rt_gene?tab=readme-ov-file)
