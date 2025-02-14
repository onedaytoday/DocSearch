# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies required for pyenv and Python builds
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PATH="/root/.pyenv/bin:${PATH}"
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PATH}"

# Install Python 3.13.0t using pyenv
RUN /bin/bash -c "source ~/.bashrc && pyenv install 3.13.0t"

# Set the global Python version
RUN /bin/bash -c "source ~/.bashrc && pyenv global 3.13.0t"

# Verify the Python version
RUN /bin/bash -c "source ~/.bashrc && pyenv rehash && python --version"

# Copy source.py into the container
COPY main.py /root/source.py

# Set the default command to run main.py using Python 3.13.0t
CMD ["python", "/root/source.py"]


