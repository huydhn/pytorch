#!/bin/bash

set -ex

install_ubuntu() {
  # NVIDIA dockers for RC releases use tag names like `11.0-cudnn9-devel-ubuntu18.04-rc`,
  # for this case we will set UBUNTU_VERSION to `18.04-rc` so that the Dockerfile could
  # find the correct image. As a result, here we have to check for
  #   "$UBUNTU_VERSION" == "18.04"*
  # instead of
  #   "$UBUNTU_VERSION" == "18.04"
  if [[ "$UBUNTU_VERSION" == "20.04"* ]]; then
    cmake3="cmake=3.16*"
  elif [[ "$UBUNTU_VERSION" == "22.04"* ]]; then
    cmake3="cmake=3.22*"
  elif [[ "$UBUNTU_VERSION" == "24.04"* ]]; then
    cmake3="cmake=3.28*"
  else
    echo "Unknown Ubuntu version $UBUNTU_VERSION"
    exit 1
  fi

  # Install common dependencies
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl
  # Install git from microsoft/git prebuilt .deb. We previously used the
  # git-core PPA via add-apt-repository, but Launchpad's API endpoint has been
  # unreliable and the distro's stock git (2.34 on jammy) is too old for
  # `git submodule update --filter=tree:0` that actions/checkout invokes.
  MS_GIT_VERSION="2.53.0.vfs.0.7"
  case "$(dpkg --print-architecture)" in
    amd64) MS_GIT_ARCH="amd64" ;;
    arm64) MS_GIT_ARCH="arm64" ;;
    *) echo "ERROR: unsupported arch $(dpkg --print-architecture) for microsoft/git" >&2; exit 1 ;;
  esac
  MS_GIT_DEB="microsoft-git_${MS_GIT_VERSION}_${MS_GIT_ARCH}.deb"
  curl -fsSL -o "/tmp/${MS_GIT_DEB}" \
    "https://github.com/microsoft/git/releases/download/v${MS_GIT_VERSION}/${MS_GIT_DEB}"
  apt-get install -y --no-install-recommends "/tmp/${MS_GIT_DEB}"
  rm -f "/tmp/${MS_GIT_DEB}"
  apt-get update
  # TODO: Some of these may not be necessary
  deploy_deps="libffi-dev libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libgdbm-dev libsqlite3-dev uuid-dev tk-dev"
  numpy_deps="gfortran"
  apt-get install -y --no-install-recommends \
    $numpy_deps \
    ${deploy_deps} \
    ${cmake3} \
    apt-transport-https \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    curl \
    libatlas-base-dev \
    libc6-dbg \
    libyaml-dev \
    libz-dev \
    libjemalloc2 \
    libgl1 \
    libjpeg-dev \
    libasound2-dev \
    libsndfile-dev \
    libssl-dev \
    software-properties-common \
    wget \
    sudo \
    vim \
    jq \
    libtool \
    vim \
    unzip \
    gpg-agent \
    gdb \
    bc \
    zip \
    valgrind

  # Should resolve issues related to various apt package repository cert issues
  # see: https://github.com/pytorch/pytorch/issues/65931
  apt-get install -y libgnutls30

  # Hard-fail if git is missing or older than 2.36 (the minimum for
  # `git submodule update --filter=tree:0`, which actions/checkout invokes).
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git was not installed" >&2
    exit 1
  fi
  GIT_VERSION=$(git --version | awk '{print $3}')
  GIT_MAJOR=${GIT_VERSION%%.*}
  GIT_MINOR=${GIT_VERSION#*.}; GIT_MINOR=${GIT_MINOR%%.*}
  if (( GIT_MAJOR < 2 || (GIT_MAJOR == 2 && GIT_MINOR < 36) )); then
    echo "ERROR: git ${GIT_VERSION} is too old; need >= 2.36" >&2
    exit 1
  fi

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
