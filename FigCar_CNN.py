
# coding: utf-8

# In[2]:


import tensorflow as tf
import scipy.io
import keras
import os


# In[3]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import h5py 
from glob import glob


# In[4]:


a = tf.truncated_normal([16,128,128,3])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.shape(a))


# In[5]:


b=tf.reshape(a,[16,49152])
sess.run(tf.shape(b))


# In[21]:


import scipy.io as spio
def group_images(path):
    """Groups and renames images per car model in appropriate folders specified by path"""
    
    path2 = path + 'Train/'
    path3 = path + 'Validation/'
    path4 = path + 'Test/'
    if not os.path.exists(path4):
        os.makedirs(path4)
    test_tar_location = path + 'cars_test/'
    if os.path.exists(test_tar_location):
        os.rename(test_tar_location, path + 'Test/cars_test/')
    """Verifies path creation"""
    print("Created Test folder and successfully stored test images")
    
    mat = scipy.io.loadmat(path + 'cars_annos.mat')
    annotations = mat['annotations']
    class_names = mat['class_names']
    print("Instantiated DB")
    
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

file_location = input("Location of Data ")
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

