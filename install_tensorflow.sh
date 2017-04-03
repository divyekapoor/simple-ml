#!/bin/bash
# Install tensorflow on Ubuntu.
# Tested with Ubuntu 16.10

set -e
set -x

sudo apt-get install -y python-numpy python-dev python-pip python-wheel python3-numpy python3-dev python3-pip python3-wheel

# For GPU with Nvidia
# sudo apt-get install libcupti-dev

export INSTALL_BAZEL=
which bazel || export INSTALL_BAZEL="true"

if [[ -n "${INSTALL_BAZEL}" ]]; then
  echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
  curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
  sudo apt-get update && sudo apt-get install bazel
fi

git submodule update
cd tensorflow
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

source venv/bin/activate
pip3 install /tmp/tensorflow_pkg/*.whl
