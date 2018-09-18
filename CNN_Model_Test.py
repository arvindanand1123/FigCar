
# coding: utf-8

<<<<<<< HEAD
# In[7]:
=======
# In[33]:
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf


import tensorflow as tf
import scipy.io
import keras
import os
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import h5py 
from glob import glob
import scipy.io as spio
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


<<<<<<< HEAD
# In[8]:
=======
# In[35]:
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf


def load_dataset(path):
    data = load_files(path)
    car_files = np.array(data['filenames'])
    car_targets = np_utils.to_categorical(np.array(data['target']), 189)
    return car_files, car_targets
<<<<<<< HEAD
file_location = '/home/arvind_anand1123/Workspace/FigCar/Input/'
=======
file_location = '/home/arvind_anand1123/Workspace/FigCar/Input'
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf
test_files, test_targets = load_dataset(file_location)
print('There are %d test car images.'% len(test_files))


<<<<<<< HEAD
# In[9]:
=======
# In[36]:
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf


def path_to_tensor(img_path):     
    # loads RGB image as PIL.Image.Image type     
<<<<<<< HEAD
    img = image.load_img(img_path, target_size=(124, 124))
=======
    img = image.load_img(img_path, target_size=(224, 224))
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf
    img = img.convert('1')
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)     
    x = image.img_to_array(img)     
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor     
    return np.expand_dims(x, axis=0)  
def paths_to_tensor(img_paths):     
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]     
    return np.vstack(list_of_tensors)


<<<<<<< HEAD
# In[10]:
=======
# In[37]:
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf


test_tensor = paths_to_tensor(test_files).astype('float32')/255


<<<<<<< HEAD
# In[11]:
=======
# In[38]:
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf


def CNN():
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential
<<<<<<< HEAD
    from keras.backend import clear_session

    clear_session()

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                            input_shape=(124, 124, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(189, activation='softmax'))

=======

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(9,9), input_shape=(224, 224, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(7,7)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(300))
    model.add(Activation("relu"))
    model.add(Dense(189))
    model.add(Activation("softmax"))
>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    
    model.load_weights('/home/arvind_anand1123/Workspace/FigCar/saved_models/weights.best.from_scratch.hdf5')
    
    # get index of predicted car model for each image in test set
    car_model_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensor]
    mat = scipy.io.loadmat('/home/arvind_anand1123/Data/cars_annos.mat')
    class_names = mat['class_names']
    var = (class_names[0][car_model_predictions[0]][0])
    print(var)

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(car_model_predictions)==np.argmax(test_targets, axis=1))/len(car_model_predictions)
    accur = ('Test accuracy: %.4f%%' % test_accuracy)
<<<<<<< HEAD
    return var + '\n' + accur    
    


# In[13]:
=======
    return var + '\n' + accur
def VGG():
    from keras import applications
    from keras.preprocessing.image import ImageDataGenerator
    from keras import optimizers
    from keras.models import Sequential, Model 
    from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
    from keras import backend as k 
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
    
    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
    
    #Freeze layers
    for layer in model.layers[:5]:
        layer.trainable = False

    #Custom layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(189, activation="softmax")(x)

    model_final = Model(input = model.input, output = predictions)

    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    
    model_final.load_weights('home/arvind_anand1123/Workspace/FigCar/saved_models/vgg16_1.h5')
    
    test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)
    
    

>>>>>>> 65c44d251f2512e56aa14814705f1f0df70aa6bf
