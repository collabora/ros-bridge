#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SUFFIX="3"

# for map server
if [ "$ROS_VERSION" = "2" ]; then
    ADDITIONAL_PACKAGES="ros-$ROS_DISTRO-navigation2
                         ros-$ROS_DISTRO-nav2-bringup"
else
    ADDITIONAL_PACKAGES="ros-$ROS_DISTRO-map-server
                         ros-$ROS_DISTRO-ros-numpy"
fi

echo ADDITIONAL PACKAGES $ADDITIONAL_PACKAGES

sudo apt update
sudo apt-get install --no-install-recommends -y \
    python3-pip \
    python3-rosinstall \
    python3-osrf-pycommon \
    python3-catkin-tools \
    python3-catkin-pkg \
    python3-catkin-pkg-modules \
    python3-rosdep \
    python3-wstool \
    python3-opencv \
    ros-$ROS_DISTRO-derived-object-msgs \
    ros-$ROS_DISTRO-rosbridge-suite \
    ros-$ROS_DISTRO-cv-bridge \
    python3-tk \
    wget \
    qt5-default \
    build-essential \
    python-is-python3 \
    $ADDITIONAL_PACKAGES

python -m pip install --upgrade pip
python -m pip install -r $SCRIPT_DIR/requirements.txt
if [ "$CUDA_VER" = "cu112" ]; then
    gdown --fuzzy https://drive.google.com/file/d/1ewbGql6WqopBZXJDnKFsTUN8yrAUqMaX/view?usp=sharing -O $SCRIPT_DIR/
    python -m pip install torchvision==0.11.2
    python -m pip install $SCRIPT_DIR/torch-1.10.0a0+git302ee7b-cp38-cp38-linux_x86_64.whl
else
    python -m pip install torch==1.10.1+$CUDA_VER torchvision==0.11.2+$CUDA_VER torchaudio==0.10.1 -f https://download.pytorch.org/whl/$CUDA_VER/torch_stable.html
fi
python -m pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/$CUDA_VER/torch1.10.1/index.html
python -m pip install mmdet==2.20
python -m pip install --upgrade pyOpenSSL
