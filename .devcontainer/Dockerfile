FROM mcr.microsoft.com/vscode/devcontainers/javascript-node:0-20
# Install dependencies for pyenv and graphviz
RUN apt-get update && \
    apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev git graphviz

# Install pyenv
RUN curl https://pyenv.run | bash

# Set up pyenv environment
RUN export PATH="/root/.pyenv/bin:$PATH" && \
    eval "$(pyenv init --path)" && \
    eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)"

# Install Python 3.9.7 and set it as global
RUN export PATH="/root/.pyenv/bin:$PATH" && \
    eval "$(pyenv init --path)" && \
    eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)" && \
    pyenv install 3.9.7 && \
    pyenv global 3.9.7

RUN apt-get install -y git-lfs && \
    git lfs install

# Upgrade pip and install pydot, tensorflow, and tensorflow[wizard]
RUN apt install python3-pip -y && \
    pip install --upgrade pip && \
    pip install pydot tensorflow tensorflow[wizard] seaborn matplotlib