## Download and install PyCharm with Anaconda plugin
https://www.jetbrains.com/pycharm/promo/anaconda/

### Some modifications.
Modification 1 and 2 is due to SIP in MacOS, that SIP protects some system locations from visiting. \
This is a direct link to let the tensorflow_gpu know where to find the cuda lib. \
Within the tensorflow_gpu vir env. \
In terminal.
#### 1
```
install_name_tool -change @rpath/libcusolver.10.0.dylib /usr/local/cuda/lib/libcusolver.10.0.dylib -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib -change @rpath/libcublas.10.0.dylib /usr/local/cuda/lib/libcublas.10.0.dylib /Users/hl/anaconda3/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
```
#### 2
```
install_name_tool -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib -change @rpath/libcublas.10.0.dylib /usr/local/cuda/lib/libcublas.10.0.dylib -change @rpath/libcudnn.7.dylib /usr/local/cuda/lib/libcudnn.7.dylib -change @rpath/libcufft.10.0.dylib /usr/local/cuda/lib/libcufft.10.0.dylib -change @rpath/libcurand.10.0.dylib /usr/local/cuda/lib/libcurand.10.0.dylib -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib /Users/hl/anaconda3/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so
```
#### 3
In the PyCharm, "Run", "Edit configurations". \
Add env var LD_LIBRARY_PATH.
```
LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
```

### Run this sample
```
import tensorflow as tf
print ("TensorFlow version: " + tf.__version__)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')
X_train /= 255
X_test /= 255
n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)) )
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tensor_board = tf.keras.callbacks.TensorBoard('./logs/LeNet-MNIST-1')

model.fit(X_train, y_train, batch_size=128, epochs=2, verbose=1, validation_data=(X_test,y_test), callbacks=[tensor_board])
```
