import matplotlib.pyplot as plt #for graph plotting
import matplotlib.image as mpimg #for processing image
import os,time #to compute execution time
import numpy as np 
import csv #to use csv format
import cv2; # to read images
from scipy.misc import imresize #to use resize functions. It is not used in the coding
import h5py # to store learned weights in binary formats

from keras.applications.imagenet_utils import preprocess_input, decode_predictions  #not used in project
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for augmenting image.
#ImageDataGenerator- for augmenting images in folder.
# array_to_img, img_to_array, load_img - used for augmenting individual images. Not used in Project
from keras.models import Sequential # two models-Sequential-arranges in sequential manner and stack processing, so easier processing , Functional API-Creates complex structure
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K #backend for keras- Theano, Tensorflow. In project Tensorflow is used

start = time.time()
num_channel=1 
img_width, img_height = 150, 150  #original image transferred and already stored

train_data_dir = './Udata/train'   #train using all images in the folder
validation_data_dir = './Udata/test' #testing using all images in the folder
test_data_dir='./test/' #not needed
nb_train_samples = 65 #65 images
nb_validation_samples = 19 #19 images
epochs = 50 #can be changed
batch_size = 16 #for augmenting image and creating more image if dataset is small. It specifies how many images must be created from 1 image. Maximum 128

#Design part

if K.image_data_format() == 'channels_first':  #automatically decides whether to use theano or tensorflow
    input_shape = (3, img_width, img_height) #for theano
else:
    input_shape = (img_width, img_height, 3) #tensorflow

model = Sequential() 

#first layer
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) #convolution layer. 32- filters for 32 features. (3,3)- one filter matrix size. The sizes may be (1,1) or (3,3), (5,5). It specifies the window size based on which img reduction takes place
#in convolution layer default striding size- (1) by which filter is moved. After conv size of image(150,150) reduced bcoz of striding
model.add(Activation('relu'))  # used for converting linear to non-linear(has only +ve values)
#In RELU if pixel value>0 then keep it 1-white otherwise 0-black
model.add(MaxPooling2D(pool_size=(2, 2))) #default stride size-(2). Window size is (2,2). It can be anything
#after maxpooling size of image will be reduced bcoz of striding

#second layer
model.add(Conv2D(32, (3, 3))) #32 features is userdefined. if we increase it then accuracy will increase
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Third layer
model.add(Conv2D(64, (3, 3))) # to increase accuracy in last layer it is set as 64. It can be changed.
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#finally the of image after 3 layers will be (17,17,64)

#fully connected layer
model.add(Flatten()) # convert image(17,17,64) into 1D(17*17*64 =18496) features
model.add(Dense(64)) #64 represents the no. of neurons. 18496 features converted into 64 features. We can give aany value instead of 64. Tested for 12,32 and 64
model.add(Activation('relu'))
model.add(Dropout(0.5)) #disable 50% of neurons to make independent of neurons, fast processing, avoid overfitting
model.add(Dense(1))#one neuron for prediction since binary prediction. for multiclass no. of neurons=no.of class labels
model.add(Activation('sigmoid')) #must be sigmoid for binary classification. for multiclass use softmax

# Implementation Part

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']) #start compilation process. binary_crossentropy bcoz binary classification
			  
#categorical_crossentropy for multiclass , rmsprop-rootmeansquare propagation.we have many optimizer which is used for updating the weight			  

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True) #create a datagenerator for augmenting the image by specifying the scale, shear, zoom,flip. We can add more transformations for generating new image from existing image

test_datagen = ImageDataGenerator(rescale=1. / 255) #only normalize the original image for testing. No other transformations must be done

#apply the created datagenerator to the image in training set to create new images. 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary') #class_mode= categorical for multiclass
	
	
	
#apply the created datagenerator to the image in testing set to create new images
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#starts training and testing. 
hist=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size, #no. of training samples used for one iteration
    epochs=epochs, #no. of epochs=no of training
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)# each training will be followed by one testing in one epoch

# evaluating the model ie testing is evaluated
scoreSeg = model.evaluate_generator(validation_generator)#returns an array with loss and accuracy
print("Accuracy = ",scoreSeg[1]*100) # in percentage
print("Loss=",scoreSeg[0])
predict = model.predict_generator(validation_generator) #contains the class label predicted for each image in test folder

#testing individual image
test_image = cv2.imread('test/Healthy/Im036_0.jpg')
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
if num_channel==1:
	if K.image_dim_ordering()=='th': #theano
		test_image= np.expand_dims(test_image, axis=0) #theano and tensorflow are 4D
		#test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape) #display shape of image with dimensions
	else: #
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
print('prediction of test image')
print('-------------------------')

#usage of predict function
pred=model.predict(test_image) #predict returns a value between 0 to 1
print(pred)
pred[pred >= 0.5]=1
pred[pred < 0.5]=0



print(model.predict(test_image))
print("Label of test image")
print('-------')

#usage predict_classes function
classes=model.predict_classes(test_image)
print(model.predict_classes(test_image)) #predict_classes returns the exact class label
train_generator.class_indices
print("Label Index")
print(train_generator.class_indices)# returns all the class label and corresponding index
if pred== 1:
  prediction = 'UnHealthy'
else:
  prediction = 'Healthy'

print(prediction)
image = mpimg.imread("test/Healthy/Im036_0.jpg")
plt.title(prediction)
plt.imshow(image)  #add image to graph
plt.show() #graph display with dimension

#hist instance of fit_generator used for training and testing
#it contains info about training loss , testing loss, training accuracy and testing accuracy
plt.plot(hist.history['acc'])#history used to retrieve loss and accuracy. acc-training accuracy, loss- loss  
plt.plot(hist.history['val_acc'])#val_acc- testing accuracy, val_loss -testing loss

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

end=time.time() - start
day = end // (24 * 3600)
end = end % (24 * 3600)
hour = end // 3600
end %= 3600
minutes = end // 60
end %= 60
seconds = end
print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))