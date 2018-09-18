
# coding: utf-8

# In[7]:


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


# In[8]:


def load_dataset(path):
    data = load_files(path)
    car_files = np.array(data['filenames'])
    car_targets = np_utils.to_categorical(np.array(data['target']), 189)
    return car_files, car_targets
file_location = '/home/arvind_anand1123/Workspace/FigCar/Input/'
test_files, test_targets = load_dataset(file_location)
print('There are %d test car images.'% len(test_files))


# In[9]:


def path_to_tensor(img_path):     
    # loads RGB image as PIL.Image.Image type     
    img = image.load_img(img_path, target_size=(124, 124))
    img = img.convert('1')
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)     
    x = image.img_to_array(img)     
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor     
    return np.expand_dims(x, axis=0)  
def paths_to_tensor(img_paths):     
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]     
    return np.vstack(list_of_tensors)


# In[10]:


test_tensor = paths_to_tensor(test_files).astype('float32')/255


# In[11]:


def CNN():
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential
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
    return var + '\n' + accur    
    


# In[13]:
