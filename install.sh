#!/bin/bash
# Install requirements.

virtualenv -p python3 venv
source venv/bin/activate

./install_tensorflow.sh
pip3 install -U -r requirements.txt
