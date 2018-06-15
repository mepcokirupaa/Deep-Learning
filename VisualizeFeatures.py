import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os,time
import numpy as np
import csv
import cv2;
from scipy.misc import imresize
import h5py

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

start = time.time()
num_channel=1
img_width, img_height = 150, 150
train_data_dir = './Udata/train' #after sepearating the images as test and train
validation_data_dir = './Udata/test'
test_data_dir='./test/'
nb_train_samples = 65
nb_validation_samples = 19
epochs =5
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height) #  for width*heigth*depth(rgb). depth=3 for colored image(r,g,b). if gray scale then depth=1
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
conv1=model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.summary() # display the shape of each layer(4 layers- includeing fully connected). Displays output of each layer including parameters, features

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save_weights('./models/weights.h5') #to store final learned updated weights in binary format. It can be used for measuring the performance. Sensitivity, Specificity etc..
model.save('./models/model.h5') # to store the model
testdata_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
scoreSeg = model.evaluate_generator(validation_generator)
print("Accuracy = ",scoreSeg[1]*100)
print("Loss=",scoreSeg[0])
predict = model.predict_generator(testdata_generator)


#display intermediate values for one test image

test_image = cv2.imread('Udata/test/Unhealthy/Im053_1.jpg')
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
print("-------------------1st Layer----------------------")
print("---------------------------------------------------")
print("------------------After Convolution----------------------")
def get_featuremaps(model,layer_idx,X_batch):
    get_activations=K.function([model.layers[0].input,K.learning_phase()],[model.layers[layer_idx].output,])
    activations=get_activations([X_batch,0])
    return activations

	
# to take intermediate result change the value of layer_num and filter_num. layer_num=filter_num
layer_num=0 #to display convoluted image.layer_num can take values 0 to 8. if 0 it represents first convolution layer, 1- first Activation, 2- first maxpoolong
filter_num=0 # to display feature map.filter_num can take values between 0 to 8 which represents the feature map generated in each layer. 
activations=get_featuremaps(model,int(layer_num),test_image)
print(np.shape(activations))
feature_maps=activations[0][0]
print(np.shape(feature_maps))

fig=plt.figure(figsize=(5,5))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("fmap-layer{}".format(layer_num)+"-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(5,5))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax=fig.add_subplot(subplot_num,subplot_num,i+1)
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("fmap-layer-{}".format(layer_num)+'.jpg')

print("-----------------------------------------------------------")
print("------------------After Activation-------------------------")

layer_num=1
filter_num=1

activations=get_featuremaps(model,int(layer_num),test_image)
print(np.shape(activations))
feature_maps=activations[0][0]
print(np.shape(feature_maps))

fig=plt.figure(figsize=(5,5))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("fmap-layer{}".format(layer_num)+"-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(5,5))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax=fig.add_subplot(subplot_num,subplot_num,i+1)
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("fmap-layer-{}".format(layer_num)+'.jpg')

print("------------------------------------------------------------")
print("--------------------After MaxPooling------------------------")

layer_num=2
filter_num=2

activations=get_featuremaps(model,int(layer_num),test_image)
print(np.shape(activations))
feature_maps=activations[0][0]
print(np.shape(feature_maps))

fig=plt.figure(figsize=(5,5))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("fmap-layer{}".format(layer_num)+"-filternum-{}".format(filter_num)+'.jpg')


num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(5,5))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax=fig.add_subplot(subplot_num,subplot_num,i+1)
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("fmap-layer-{}".format(layer_num)+'.jpg')

end=time.time() - start
day = end // (24 * 3600)
end = end % (24 * 3600)
hour = end // 3600
end %= 3600
minutes = end // 60
end %= 60
seconds = end
print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))