# Jupyter notebook tensorflow-gpu kernel
Reference \
https://www.pugetsystems.com/labs/hpc/The-Best-Way-to-Install-TensorFlow-with-GPU-Support-on-Windows-10-Without-Installing-CUDA-1187/

This is used after install of tensorflow-gpu, making it a jupyter notebook kernel and can be used for notebook.

Within conda virtual env where tensorflow_gpu was installed (with the tensorflow_gpu environment activated).
```
conda install jupyter
conda install ipykernel
python -m ipykernel install --user --name tensorflow_gpu --display-name "TensorFlow-GPU"   # tensorflow_gpu is the virtual env.
```
Then open a jupyter notebook.
```
jupyter notebook
```
From the 'New' drop-down menu select the 'TensorFlow-GPU' kernel that was added, then start coding.

## MNIST example
Following are Python snippets that can be copied into cells in the Jupyter notebook to setup and train LeNet-5 with MNIST digits data.

```
import tensorflow as tf
print ("TensorFlow version: " + tf.__version__)

# Load and process the MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')
X_train /= 255
X_test /= 255
n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

# Create the LeNet-5 neural network architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)) )
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())          
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set log data to feed to TensorBoard for visual analysis
tensor_board = tf.keras.callbacks.TensorBoard('./logs/LeNet-MNIST-1')

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1, validation_data=(X_test,y_test), callbacks=[tensor_board])
```
The results, after running that training for 15 epochs the last epoch gave.
```
Epoch 15/15
60000/60000 [==============================] - 6s 102us/step - loss: 0.0192 - acc: 0.9936 - val_loss: 0.0290 - val_acc: 0.9914
```

## Look at the job run with TensorBoard
First, shutdown notebook, ctr + C.
```
conda install bleach   # This may not need.
tensorboard --logdir=./logs --host localhost --port 6006
```
It will give an address `http://localhost:6006`, open that in browser and it will show TensorBoard.

### Thoughts
This was a model with 1.2 million training parameters and a dataset with 60,000 images. 
It took 1 minute and 26 seconds utilizing the NVIDIA GeForce 1070 in a laptop. 
For reference it took 26 minutes using all cores at 100% of the Intel 6700HQ CPU.
That's an 18 fold speedup on the GPU.

          
  
