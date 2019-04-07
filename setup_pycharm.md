## download and install PyCharm with Anaconda plugin
https://www.jetbrains.com/pycharm/promo/anaconda/

### Some modifications.
Within the tensorflow_gpu vir env. \
in terminal.
#### 1
```
install_name_tool -change @rpath/libcusolver.10.0.dylib /usr/local/cuda/lib/libcusolver.10.0.dylib -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib -change @rpath/libcublas.10.0.dylib /usr/local/cuda/lib/libcublas.10.0.dylib /Users/hl/anaconda3/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
```
#### 2
```
install_name_tool -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib -change @rpath/libcublas.10.0.dylib /usr/local/cuda/lib/libcublas.10.0.dylib -change @rpath/libcudnn.7.dylib /usr/local/cuda/lib/libcudnn.7.dylib -change @rpath/libcufft.10.0.dylib /usr/local/cuda/lib/libcufft.10.0.dylib -change @rpath/libcurand.10.0.dylib /usr/local/cuda/lib/libcurand.10.0.dylib -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib /Users/hl/anaconda3/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so
```
#### 3
Add env var.
```
LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
```
