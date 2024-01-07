ARG UBUNTU_VERSION=20.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu$UBUNTU_VERSION
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu$UBUNTU_VERSION
ENV DEBIAN_FRONTEND noninteractive
ARG CMAKE_INSTALL_PREFIX=/usr/local
ARG NUM_THREADS
ARG NODEJS_MAJOR_VERSION

# Display arguments
RUN echo " \
  Ubuntu version: ${UBUNTU_VERSION}\n \
  NodeJS version: ${NODEJS_MAJOR_VERSION}\n \
"

RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install dependencies via apt
ENV DEBCONF_NOWARNINGS yes
RUN set -x && \
  apt-get update -y -qq && \
  apt-get install -y software-properties-common && \
  add-apt-repository -y ppa:git-core/ppa && \
  apt-get upgrade -y -qq --no-install-recommends && \
  : "basic dependencies" && \
  apt-get install -y -qq \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    tar \
    unzip && \
  : "OpenCV dependencies" && \
  apt-get install -y -qq \
    libjpeg-dev \
    libpng++-dev \
    libtiff-dev \
    libopenexr-dev \
    libwebp-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev && \
  : "remove cache" && \
  apt-get autoremove -y -qq && \
  rm -rf /var/lib/apt/lists/*

# Node
 RUN mkdir -p /etc/apt/keyrings; \
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg; \
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODEJS_MAJOR_VERSION}.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list; \
   apt-get update && apt-get install -y nodejs;

ENV CPATH=${CMAKE_INSTALL_PREFIX}/include:${CPATH}
ENV C_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${CPLUS_INCLUDE_PATH}
ENV LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}

# CUDA
ENV CUDACXX /usr/local/cuda/bin/nvcc
ENV CUDA_INCLUDE_DIRS /usr/local/cuda/include
ENV CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/
