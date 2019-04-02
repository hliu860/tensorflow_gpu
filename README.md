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

## Some modification to the source
Disable SIP, in recovery mode (command+R before mac starts)
```
csrutil status
csrutil disable
```
After all done enable it in recovery mode.
```
csrutil status
csrutil enable
```

## modify ~/.bash_profile
```
export PATH="$PATH:$HOME/bin"  # This is for installing bazel from binary that bazel locates in #HOME/bin

export TMP=/c/tempdir  # This is for bazel that it somehow auto configure $TMP to c:/windows/tmp 

export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
export PATH=$DYLD_LIBRARY_PATH:$PATH
export PATH="/usr/local/cuda/bin":$PATH

export PATH="/usr/local/opt/llvm/bin:$PATH"   # This not sure if it is useful.
```

## configure
Inside tensorflow 
```
./configure
```
No to unnessary options \
Yes to CUDA and cuDNN, compute capacity is 3.0

## A preprocess
```
echo $TMP
sudo mkdir -p $TMP   # This is a workaround for bazel, bazel does not create any files in that dir during compiling.
```
## Compile
```
bazel build  --config=noaws --config=nogcp --config=nohdfs --config=noignite  --config=nokafka  --config=nonccl --verbose_failures --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```
## build wheel file and pip install
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install  /tmp/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-macosx_10_7_x86_64.whl 
```
## keep wheel file for future use
```
mv /tmp/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-macosx_10_7_x86_64.whl  ~/
