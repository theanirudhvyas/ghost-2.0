#!/bin/bash

echo "=================================================="
echo "          GHOST 2.0 Setup Script                 "
echo "=================================================="
echo ""

start_time=$(date +%s)

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed or not in PATH. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating conda environment 'ghost-new' with Python 3.10..."
# Create and activate conda environment with Python 3.10
conda create -n ghost-new python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ghost-new

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch and torchvision..."
# Install PyTorch and torchvision (specific versions for better compatibility)
pip install torch==2.2.2 torchvision==0.17.2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing face detection and alignment libraries..."
# Install face detection and alignment libraries
pip install face-alignment>=1.3.5
pip install facenet_pytorch>=2.5.2
pip install insightface==0.7.3

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up 3D modeling dependencies..."
# Add conda-forge channel for pytorch3d
conda config --add channels conda-forge
conda config --set channel_priority strict
pip install -U 'git+https://github.com/facebookresearch/fvcore'
conda install pytorch3d -c pytorch3d-nightly -y

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing additional required packages..."
# Install other requirements
pip install -r requirements.txt

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing specific packages that might be missing..."
# Install specific packages that might be missing
pip install transformers simple-lama-inpainting lightning onnxruntime==1.21.0 diffusers==0.32.2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fixing numpy version for compatibility..."
# Fix numpy version for compatibility with various packages
pip uninstall numpy -y
pip install numpy==1.26.4

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up repository structure..."
# Create repos directory if it doesn't exist
mkdir -p repos

# Clone required repositories
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cloning required repositories..."
cd repos

if [ ! -d "DECA" ]; then
    echo "Cloning DECA repository..."
    git clone https://github.com/yfeng95/DECA.git
else
    echo "DECA repository already exists, skipping..."
fi

if [ ! -d "EMOCA" ]; then
    echo "Cloning EMOCA repository..."
    git clone https://github.com/anastasia-yaschenko/EMOCA.git
else
    echo "EMOCA repository already exists, skipping..."
fi

if [ ! -d "BlazeFace_PyTorch" ]; then
    echo "Cloning BlazeFace_PyTorch repository..."
    git clone https://github.com/anastasia-yaschenko/BlazeFace_PyTorch.git
else
    echo "BlazeFace_PyTorch repository already exists, skipping..."
fi

if [ ! -d "stylematte" ]; then
    echo "Cloning stylematte repository..."
    git clone https://github.com/chroneus/stylematte.git
else
    echo "stylematte repository already exists, skipping..."
fi

cd ..

# Create necessary directories
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating necessary directories..."
mkdir -p aligner_checkpoints
mkdir -p blender_checkpoints
mkdir -p src/losses/gaze_models
mkdir -p weights

# Make run_inference.sh executable if it exists
if [ -f "run_inference.sh" ]; then
    chmod +x run_inference.sh
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Made run_inference.sh executable"
fi

# Calculate execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "=================================================="
echo "Setup completed in ${minutes}m ${seconds}s!"
echo "=================================================="
echo ""
echo "IMPORTANT: Please download the following files manually:"
echo ""
echo "1. EMOCA ResNet50 folder from:"
echo "   https://github.com/anastasia-yaschenko/emoca/releases/tag/resnet"
echo "   Place in:"
echo "   - repos/EMOCA/gdl_apps/EmotionRecognition/"
echo "   - assets/EmotionRecognition/image_based_networks/"
echo ""
echo "2. Model files:"
echo "   - aligner_1020_gaze_final.ckpt -> aligner_checkpoints/"
echo "   - blender_lama.ckpt -> blender_checkpoints/"
echo "   - backbone50_1.pth -> weights/"
echo "   - vgg19-d01eb7cb.pth -> weights/"
echo "   - segformer_B5_ce.onnx -> weights/"
echo ""
echo "3. To run the inference:"
echo "   ./run_inference.sh"
echo "=================================================="
