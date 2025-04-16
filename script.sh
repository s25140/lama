#!/bin/bash
set -e

# User configuration
ENV_DIR="inpenv"
PYTHON_PATH="/usr/bin/python3"
REQUIREMENTS_FILE="requirements.txt"

# Updated PyTorch + CUDA versions
TORCH_VERSION="2.0.1"
TORCHVISION_VERSION="0.15.2"
TORCHAUDIO_VERSION="2.0.2"
CUDA_VERSION_TAG="cu118"  # CUDA 11.8
TORCH_WHL_URL="https://download.pytorch.org/whl/torch_stable.html"

# Configure dpkg for container environment (fixes cross-device link issue)
echo "[INFO] Configuring dpkg for container environment..."
mkdir -p /etc/dpkg/dpkg.cfg.d/
echo "force-unsafe-io" > /etc/dpkg/dpkg.cfg.d/docker-apt-speedup
echo "no-debsig" >> /etc/dpkg/dpkg.cfg.d/docker-apt-speedup
echo "path-exclude=/usr/share/doc/*" > /etc/dpkg/dpkg.cfg.d/exclude-docs

# Install system dependencies
echo "[INFO] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    gnupg2

# Add NVIDIA repository and install CUDA toolkit only (no drivers)
echo "[INFO] Setting up NVIDIA CUDA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
# Container-compatible CUDA toolkit installation (no drivers)
apt-get install -y --no-install-recommends cuda-toolkit-11-8

# Remove existing environment if it exists
if [ -d "$ENV_DIR" ]; then
    echo "[INFO] Removing existing virtual environment..."
    rm -rf $ENV_DIR
fi

echo "[INFO] Creating virtual environment..."
$PYTHON_PATH -m pip install --upgrade pip
$PYTHON_PATH -m pip install virtualenv
$PYTHON_PATH -m virtualenv $ENV_DIR --python=$PYTHON_PATH

echo "[INFO] Activating virtual environment..."
source $ENV_DIR/bin/activate

# Ensure pip is up to date in the virtual environment
python -m pip install --upgrade pip

# Install PyTorch first
echo "[INFO] Installing PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION_TAG..."
pip install torch==${TORCH_VERSION}+${CUDA_VERSION_TAG} \
           torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG} \
           torchaudio==${TORCHAUDIO_VERSION}+${CUDA_VERSION_TAG} \
           -f $TORCH_WHL_URL

# Verify PyTorch installation and GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"

# Install updated core dependencies
echo "[INFO] Installing core dependencies..."
pip install protobuf==3.20.3
pip install hydra-core==1.3.2 
pip install pytorch-lightning==2.0.9
pip install numpy==1.24.3

# Now install from requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "[INFO] Installing packages from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE
else
    echo "[WARN] $REQUIREMENTS_FILE not found, skipping requirements installation."
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TORCH_HOME=$(pwd)
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

echo "[INFO] Setup completed successfully!"
echo "[INFO] Environment variables set:"
echo "PYTHONPATH=$PYTHONPATH"
echo "TORCH_HOME=$TORCH_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""
echo "To activate the environment, run:"
echo "source $ENV_DIR/bin/activate"

# Add persistent environment variables
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> $ENV_DIR/bin/activate
echo 'export TORCH_HOME=$(pwd)' >> $ENV_DIR/bin/activate
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"' >> $ENV_DIR/bin/activate