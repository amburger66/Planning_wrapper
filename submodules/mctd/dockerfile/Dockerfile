FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Korea/Seoul
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev cmake \
  x11-xserver-utils xvfb \
  && apt-get clean
RUN pip3 install --upgrade pip

# Install libraries
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# RUN pip3 install torch~=2.0.0
# RUN pip3 install torchvision~=0.15.1
RUN pip3 install lightning~=2.1.2
RUN pip3 install wandb~=0.17.0
RUN pip3 install hydra-core~=1.3.2
RUN pip3 install omegaconf~=2.3.0
RUN pip3 install torchmetrics[image]==0.11.4
RUN pip3 install wandb-osh==1.2.1
RUN pip3 install gluonts[torch]==0.13.1
RUN pip3 install colorama
RUN pip3 install tqdm
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install click
RUN pip3 install moviepy
RUN pip3 install imageio
RUN pip3 install einops
RUN pip3 install pandas
RUN pip3 install pyzmq
RUN pip3 install pyrealsense2
RUN pip3 install internetarchive

# From extra requirements
RUN pip3 install cython==0.29.37
RUN pip3 install setuptools==65.5.0 
RUN pip3 install wheel==0.40.0
RUN pip3 install d4rl==1.1
RUN pip3 install shimmy>=0.2.1
RUN pip3 install stable-baselines3
RUN pip3 install pip==21.0.1
RUN pip3 install gym==0.19.0
RUN pip3 install mujoco-py
RUN pip3 install rotary-embedding-torch
RUN pip3 install diffusers

RUN pip3 install shapely

# Install additional libraries
RUN apt-get update
RUN apt-get -y install libosmesa6-dev patchelf

# Upgrade torchvision to remove warning msgs
RUN pip3 install --upgrade torchvision

# # Copy the mujoco folder
# COPY ./mujoco/generated /usr/local/lib/python3.10/dist-packages/mujoco_py/generated

# # Give the authority to the user for the mujoco_py library
# RUN chmod -R 777 /usr/local/lib/python3.10/dist-packages/mujoco_py/generated

# # New User (Optional)
# ARG UNAME=jsyoon
# ARG UID=1020
# ARG GID=1020
# RUN groupadd -g $GID -o $UNAME
# RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
# USER $UNAME
# WORKDIR /home/$UNAME

# # Copy the mujoco folder
# RUN mkdir -p /home/$UNAME/.mujoco/mujoco210
# COPY ./mujoco/mujoco210 /home/$UNAME/.mujoco/mujoco210

# # Add LD_LIBRARY_PATH
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/$UNAME/.mujoco/mujoco210/bin
