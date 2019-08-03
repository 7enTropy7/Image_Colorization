import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.datasets import cifar10
from keras.layers import Dense, Input
from keras.layers import Reshape, Flatten, Conv2D, Conv2DTranspose
from keras import backend as K

(xtrain,_),(xtest,_) = cifar10.load_data()

rows = xtrain.shape[1]
cols = xtrain.shape[2]
channels = xtrain.shape[3]

xtrain_grey = np.dot(xtrain[...,:3],[0.299,0.587,0.144])
xtest_grey = np.dot(xtest[...,:3],[0.299,0.587,0.144])

xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255
xtrain_grey = xtrain_grey.astype('float32')/255
xtest_grey = xtest_grey.astype('float32')/255

xtrain = xtrain.reshape(xtrain.shape[0],rows,cols,channels)
xtest = xtest.reshape(xtest.shape[0],rows,cols,channels)
xtrain_grey = xtrain_grey.reshape(xtrain_grey.shape[0],rows,cols,1)
xtest_grey = xtest_grey.reshape(xtest_grey.shape[0],rows,cols,1)

enc_input = (rows,cols,1)
latent_dim = 256
layer_filters = [64,128,256]

inputs = Input(shape=enc_input)
x = inputs
for i in layer_filters:
    x = Conv2D(filters=i,kernel_size=3,strides=2,activation='relu',padding='same')(x)

shape = K.int_shape(x)

x = Flatten()(x)
latent = Dense(latent_dim)(x)

encoder = Model(inputs,latent)

latent_inputs = Input(shape=(latent_dim,))
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1],shape[2],shape[3]))(x)

for i in layer_filters[::-1]:
    x = Conv2DTranspose(filters=i,kernel_size=3,strides=2,activation='relu',padding='same')(x)

outputs = Conv2DTranspose(filters=channels,kernel_size=3,activation='sigmoid',padding='same')(x)

decoder = Model(latent_inputs,outputs)

autoencoder = Model(inputs,decoder(encoder(inputs)))
autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.fit(xtrain_grey,xtrain,validation_data=(xtest_grey,xtest),epochs=30,batch_size=32)

decoded_output = autoencoder.predict(xtest_grey)
imgs = decoded_output[:100]
imgs = imgs.reshape((10, 10, rows, cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('Colorized_Output/color_img.png')
plt.show()