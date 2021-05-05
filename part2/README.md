# Part 2

Comparing output with [state of the art deep network.](https://github.com/autonomousvision/connecting_the_dots)

## Setup
- Run `./setup.sh` to download datasets, model weights and to download and install required packages.

> Note: the script was written for AWS Ubuntu 18.04 Deep Learning AMI v43. Make suitable changes before running

## Run
- Run `python3 infer.py "filename"`. Filename denotes the input in `.npy` format.
- Run `python3 match.py` for stereo matching. Modify file appropriately to change input.
