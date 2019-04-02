# Tensorflow with Nvidia GPU support for MacOS
MacBook pro 15 (mid-2014) with Nvidia GT 750M 2G (GPU compute capacity 3.0)

## MacOS High Sierra 10.13.6 
Fresh install, don't install any updates.

## brew (latest)
Installl brew and update to the latest

## GPU web driver 387.10.10.10.40.105
https://github.com/vulgo/webdriver.sh 
```
brew tap vulgo/repo 
brew install webdriver.sh 
webdriver
```

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
From apple app store,change its name to Xcode9.2.app then move it to Application. \
Change to xcode 9.2 developer environment.
```
sudo xcode-select -s /Application/Xcode9.2.app
```

## bazel 0.21.0
From bazel binary, tensorflow 1.13.1 needs bazel 0.12.0 to compile.

## Anaconda python 3.6.8
Download and install anaconda then make a virtual environment with python 3.6.8

## Tensorflow 1.13.1
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git status
git checkout r1.13
```

## configure
Inside tensorflow 
```
./configure
```
No to unnessary options and yes to CUDA and cuDNN, compute capacity is 3.0

## Some modification to the source
Disable SIP, in recovery mode (command+R before mac starts)
```
csrutil status
csrutil disable
```
After all done enable it.

## Compile
```
bazel build  --config=noaws --config=nogcp --config=nohdfs --config=noignite  --config=nokafka  --config=nonccl --verbose_failures --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```
