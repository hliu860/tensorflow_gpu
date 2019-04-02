# Tensorflow with Nvidia GPU support for MacOS
MacBook pro 15 (mid-2014) with Nvidia GT 750M 2G (GPU compute capacity 3.0)

## MacOS High Sierra 10.13.6 
Fresh install, no any updates.

## brew (latest)
installl brew and update to the latest

## GPU web driver 387.10.10.10.40.105
https://github.com/vulgo/webdriver.sh \
brew tap vulgo/repo \
brew install webdriver.sh \
webdriver

## CUDA driver 410.130
Download from nvidia website. \
https://www.nvidia.com/object/mac-driver-archive.html 

## CUDA Toolkit 10.0
https://developer.nvidia.com/cuda-toolkit-archive

## cuDNN 7.4.1
Need register on the Nvidia web then download.

## nccl 2.4.2  
(not configured, not used for signle GPU but will be useful for multiple GPUs)
Need register on the Nvidia web then download.

## Xcode 9.2
From apply app store.

## bazel 0.21.0
From bazel binary.

## Anaconda python 3.6.8
Download and install anaconda then make a virtual environment with python 3.6

## Tensorflow 1.13.1
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git status
git checkout r1.13
```
