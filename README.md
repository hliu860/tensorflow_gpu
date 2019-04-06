# Tensorflow with Nvidia GPU support for MacOS
MacBook pro 15 (mid-2014) with Nvidia GT 750M 2G (GPU compute capacity 3.0)

## MacOS High Sierra 10.13.6 
### Build 17G65 from the 22m installer inside MacOS HS.
Fresh install, don't install any updates.

## brew (latest)
Installl brew and update to the latest.
```
brew update
brew upgrade
brew install coreutils
brew install swig  # This one not sure its function.
```

## GPU web driver 387.10.10.10.40.105
https://github.com/vulgo/webdriver.sh 
```
brew tap vulgo/repo 
brew install webdriver.sh 
webdriver
```

## CUDA driver 410.130
Download and install. \
https://www.nvidia.com/object/mac-driver-archive.html 

## CUDA Toolkit 10.0
Download and install. \
https://developer.nvidia.com/cuda-toolkit-archive

## cuDNN 7.4.1
https://developer.nvidia.com/rdp/form/cudnn-download-survey \
Need register on the Nvidia web then download, \
then move them to `/usr/local/cuda` which is the main cuda directory.
```
tar zxvf ~/Downloads/cudnn-10.0-osx-x64-v7.4.1.5.tar
sudo mv -v cuda/lib/libcudnn* /usr/local/cuda/lib
sudo mv -v cuda/include/cudnn.h /usr/local/cuda/include
```

## nccl 2.4.2  
(not configured, not used for signle GPU but will be useful for multiple GPUs) \
Need register on the Nvidia web then download and move to `/usr/local/`
```
xz -d nccl_2.4.2-1+cuda10.0_x86_64.txz
tar xvf nccl_2.4.2-1+cuda10.0_x86_64.tar

cd nccl_2.1.15-1+cuda9.1_x86_64
sudo mkdir -p /usr/local/nccl
sudo mv * /usr/local/nccl
sudo mkdir -p /usr/local/include/third_party/nccl
sudo ln -s /usr/local/nccl/include/nccl.h /usr/local/include/third_party/nccl
```

## Xcode 9.2
Download from the apple app store, change its name to Xcode9.2.app then move it to Application. \
Accept license and change to xcode 9.2 developer environment.
```
sudo xcodebuild -license accept  # accept the license 
sudo xcode-select -s /Application/Xcode9.2.app
```

## Compile CUDA samples
Compile some CUDA samples to check if the GPU is correctly recognized and supported.
```
cd /Developer/NVIDIA/CUDA-10.0/samples
make -C 1_Utilities/deviceQuery
./bin/x86_64/darwin/release/deviceQuery
```

## bazel 0.21.0
From bazel binary, tensorflow 1.13.1 needs bazel 0.21.0 to compile.
```
chmod +x bazel-0.21.0-installer-darwin-x86_64.sh
./bazel-<version>-installer-darwin-x86_64.sh --user
```
Use brew will install lastest bazel but it didn't work to compile tensorflow 1.13.1.
```
(brew install bazel) 
```

## Anaconda python 3.6.8
Download and install anaconda then make a virtual environment with python 3.6.8 \
Within python 3.6.8 virtual env, do following:
```
pip install -U pip six numpy wheel setuptools mock   # -U is upgrade.
pip install -U keras_applications==1.0.6 --no-deps
pip install -U keras_preprocessing==1.0.5 --no-deps
```
Reference
https://www.tensorflow.org/install/source

## Tensorflow 1.13.1
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git status
git checkout r1.13
```

## Some modification to the source
### 1
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
### 2
Remove all `__align(sizeof(T))__` from following 3 files: 
```
tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc 
tensorflow/core/kernels/split_lib_gpu.cu.cc 
tensorflow/core/kernels/concat_lib_gpu.impl.cu.cc 
```
For example:
```
extern shared __align(sizeof(T))__ unsigned char smem[];
```
becomes
```
extern shared unsigned char smem[];
```
### 3
Comment out 
```
linkopts = [“-lgomp”]
```
in 
```
tensorflow/third_party/gpus/cuda/BUILD.tpl
```
Reference:
https://github.com/tensorflow/tensorflow/issues/15172

## Modify ~/.bash_profile
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

## Configure
Inside tensorflow 
```
./configure
```
No to unnessary options \
Yes to CUDA and cuDNN, compute capacity is 3.0

## A workaround for bazel
```
echo $TMP
sudo mkdir -p $TMP   # This is a workaround for bazel, bazel does not create any files in that dir during compiling.
```
## Compile
It took more than 1 hour to compile, about 90 mins.
```
bazel build  --config=noaws --config=nogcp --config=nohdfs --config=noignite  --config=nokafka  --config=nonccl --verbose_failures --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```
## Build wheel file and pip install
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install  /tmp/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-macosx_10_7_x86_64.whl 
```
## Keep wheel file for future use
```
mv /tmp/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-macosx_10_7_x86_64.whl  ~
```
## Test tensorflow inside python
```
import tensorflow as tf

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))
```
It should give a matrix in the end.
```
[[ 22.  28.]
 [ 49.  64.]]
```

# References
https://gist.github.com/smitshilu/53cf9ff0fd6cdb64cca69a7e2827ed0f

https://gist.github.com/ageitgey/819a51afa4613649bd18

https://gist.github.com/Willian-Zhang/a3bd10da2d8b343875f3862b2a62eb3b
