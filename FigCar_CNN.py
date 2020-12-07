
# coding: utf-8

# In[ ]:


import tensorflow as tf
import scipy.io
import keras
import os


# In[ ]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import h5py 
from glob import glob


# In[ ]:


a = tf.truncated_normal([16,128,128,3])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.shape(a))


# In[ ]:


b=tf.reshape(a,[16,49152])
sess.run(tf.shape(b))


# In[ ]:


import scipy.io as spio
def group_images(path):
    """Groups and renames images per car model in appropriate folders specified by path"""
    
    path2 = path + 'Train/'
    path3 = path + 'Validation/'
    path4 = path + 'Test/'
    if not os.path.exists(path4):
        os.makedirs(path4)
    test_tar_location = path + 'cars_test/'
    
    mat = scipy.io.loadmat(path + 'cars_annos.mat')
    annotations = mat['annotations']
    class_names = mat['class_names']
    print("Instantiated DB")
    testMat = scipy.io.loadmat(path + 'cars_test_annos.mat')['annotations']
    print("Instantiated Test DB")
    
    x = 0
    for i in range((len(annotations[0]))):
        car_name = class_names[0][annotations[0][i][5][0][0]-1][0]
        car_name = car_name[:-5].replace(" ", "_")
        if car_name == 'Ram_C/V_Cargo_Van_Minivan':
            car_name = 'Ram_C_MiniVan'
        newpath = path2 + car_name
        validation_path = path3 + car_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            os.makedirs(validation_path)
            x = 0
        src = path + annotations[0][i][0][0]
        x = x + 1
        try:
            if(x == 1 or x == 2):
                os.rename(src, validation_path + '/'+ car_name + str(x) + '.jpg')
            else:
                os.rename(src, newpath + '/'+ car_name + str(x-2) + '.jpg')
        except:
            pass
    print("Train and Validation created successfully")
    

    """Test Creation"""
    newpath = '/home/arvind_anand1123/Data/Test/0/'
    for i in range((len(testMat[0]))):
        if((not os.path.exists(newpath)) and (i == 0 or i % 63 == 0)):
            newpath = path4 + str(i) + '/'
            os.makedirs(newpath)
        try:
            os.rename(test_tar_location + testMat[0][i][4][0], newpath + testMat[0][i][4][0])
        except:
            pass
    print("Test created successfully")
    

file_location = '/home/arvind_anand1123/Data/'
group_images(file_location)


# In[ ]:


def load_dataset(path):
    data = load_files(path)
    car_files = np.array(data['filenames'])
    car_targets = np_utils.to_categorical(np.array(data['target']), 189)
    return car_files, car_targets

train_files, train_targets = load_dataset(file_location + 'Train')
valid_files, valid_targets = load_dataset(file_location + 'Validation')
test_files, test_targets = load_dataset(file_location + 'Test')

car_names = [item[20:-1] for item in sorted(glob(file_location + "Train/*/"))]

print('There are %d total car categories.' % len(car_names))
print('There are %s total car images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training car images.' % len(train_files))
print('There are %d validation car images.' % len(valid_files))
print('There are %d test car images.'% len(test_files))


# In[ ]:


from keras.preprocessing import image
from tqdm import tqdm
def path_to_tensor(img_path):     
    # loads RGB image as PIL.Image.Image type     
    img = image.load_img(img_path, target_size=(224, 224))     
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)     
    x = image.img_to_array(img)     
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor     
    return np.expand_dims(x, axis=0)  
def paths_to_tensor(img_paths):     
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]     
    return np.vstack(list_of_tensors)


# In[ ]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(9,9), input_shape=(224, 224, 3)))
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
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint  
import tensorboard

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 200

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# In[ ]:


model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# In[ ]:


# get index of predicted car model for each image in test set
car_model_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(car_model_predictions)==np.argmax(test_targets, axis=1))/len(car_model_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

