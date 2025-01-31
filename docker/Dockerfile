FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common gcc
RUN apt-get install -y python3.9 python3.9-dev python3.9-distutils python3-pip python3-apt python3.9-venv
RUN apt-get update && apt-get install -y git
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

RUN /usr/bin/python3.9 -m pip install --upgrade pip

# Copy code in for installation
COPY ./ /app
WORKDIR /app

# make virtualenv 
SHELL ["/bin/bash", "-c"]
RUN \
    --mount=type=cache,target=/root/.cache/ \
    make install

# Install dependencies
# Remove code, without removing env
RUN find ./ ! -path './.venv*' ! -path './' -delete

RUN mkdir /workspace
WORKDIR /workspace

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN echo 'PS1="(d3nav) \[\]\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ \[\]"' >> /root/.bashrc
