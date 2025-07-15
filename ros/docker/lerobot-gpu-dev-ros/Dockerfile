FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
FROM osrf/ros:noetic-desktop-full


# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive
# Skip the keyboard layout selection screen
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    git git-lfs openssh-client \
    nano vim less util-linux tree \
    htop atop nvtop \
    sed gawk grep curl wget zip unzip \
    tcpdump sysstat screen tmux \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa \
    speech-dispatcher portaudio19-dev libgeos-dev \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ffmpeg build dependencies. See:
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
# TODO(aliberts): create image to build dependencies from source instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake yasm \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libunistring-dev \
    libmp3lame-dev \
    libtool \
    libvorbis-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    yasm \
    zlib1g-dev \
    nasm \
    libx264-dev \
    libx265-dev libnuma-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libopus-dev \
    libsvtav1-dev libsvtav1enc-dev libsvtav1dec-dev \
    libdav1d-dev

# Install gh cli tool
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Setup `python`
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
RUN echo 'if [ "$HOME" != "/root" ]; then ln -sf /root/.local/bin/poetry $HOME/.local/bin/poetry; fi' >> /root/.bashrc
RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.in-project true

# Setup X11
RUN apt-get update && apt-get install -y --no-install-recommends \
	libx11-dev \
	libxcursor-dev \
	libxcb1-dev \
	libxi-dev \
	libxkbcommon-dev \
	libxkbcommon-x11-dev && \
	apt clean && rm -rf /var/lib/apt/lists/*

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"


# setup ros environment
# Install apt packages
RUN apt-get update && apt-get install -y \
git nano vim wget tmux terminator curl net-tools python3-pip spacenavd libhidapi-dev \
emacs htop screen ffmpeg unzip x11-apps cmake iputils-ping net-tools python3-catkin-tools \
liboctomap-dev libfcl-dev tk-dev python3-tk && \
apt-get clean && rm -rf /var/lib/apt/lists/*  

# Install pip
RUN pip install --upgrade pip
RUN pip install dynamixel-sdk casadi rosbags opencv-python python-fcl pykin pyspacemouse

# Install ROS packages
RUN apt-get update && apt-get install -y \
  ros-noetic-joy ros-noetic-teleop-twist-joy \
  ros-noetic-moveit-msgs ros-noetic-spacenav-node \
  ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
  ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
  ros-noetic-rosserial-python ros-noetic-rosserial-client \
  ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
  ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
  ros-noetic-compressed-image-transport ros-noetic-rqt* ros-noetic-rviz \
  ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers \
  ros-noetic-dynamixel-sdk \
  ros-noetic-kdl-parser ros-noetic-kdl-parser-py \
  ros-noetic-plotjuggler-ros \
  ros-noetic-turtlebot3-msgs \
  ros-noetic-turtlebot3 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Create ROS workspace
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
# RUN mkdir -p /root/livox_ws/src
RUN mkdir -p /root/catkin_ws/src
RUN mkdir -p /root/catkin_ws/bin
RUN mkdir -p /root/catkin_ws/bag
RUN cd /root/catkin_ws && /bin/bash -c "source /opt/ros/noetic/setup.sh; catkin build"
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
# RUN echo "export ROS_IP=192.168.250.1" >> /root/.bashrc

# Copy env files
COPY screenrc /root/.screenrc

# # Install Livox
# WORKDIR /root/
# RUN git clone https://github.com/NaokiTakahashi12/Livox-SDK2.git && \
#     mkdir -p ./Livox-SDK2/build && \
#     cd ./Livox-SDK2/build && \
#     cmake .. && make -j && make install

WORKDIR /root/
