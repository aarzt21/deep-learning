#Code for ResNet50 with height and width 250

#Import Libraries
import os 
import cv2
import numpy as np
import pandas as pd
import keras
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
import time
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Dropout
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
K.set_image_data_format('channels_last')
from numpy import genfromtxt
import tensorflow as tf

#Path of image folder 
train_path  = "/content/drive/My Drive/IML Project 4 Data/food/food"
train_images = os.listdir(f'{train_path}')


#Sizes height and widt
h = 250
w = 250

#Loop  to read in the images, reorder them into in rgb format, resize and
#preprocess them 
train_data = []

for imgs in train_images:
    print(imgs)
    img = cv2.imread(f'{train_path}/{imgs}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (h, w))
    img = image.img_to_array(img)
    #Preprocess for Res-net
    img = preprocess_input(img, data_format='channels_last')
    train_data.append(img)


#Image numbers to create the triplets 
train_triplets = pd.read_csv('/content/drive/My Drive/IML Project 4 Data/train_triplets.txt', sep=" ", header=None)


#################################################################################################
#Siamete CNN model
    
#Definition of helper funtions
def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

#Sqaured or not ?
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def tripl_loss(y_true, y_pred):
        margin = K.constant(0.2)
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy2(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])


#Architecture of the Siamese CNN model with VGG16 as pre-trained weights
def get_siamese_model(input_shape, embedding_dim=128):
    
    """
        Model architecture
    """
    
    ###### Pre-Trained Res-Net
    #Look at input shapes: h, w
    print(input_shape)
    resnet_input = Input(input_shape)
    path_res = "/content/drive/My Drive/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    resnet_model = ResNet50(weights= path_res, include_top = False, input_tensor=resnet_input)
    
    for layer in resnet_model.layers:
        layer.trainable = False  
    

    net = resnet_model.output
    
    #3 Hidden Layer with d=128 for embedding
    net = Flatten(name='flatten')(net)  
    net = Dense(embedding_dim, activation='relu', name='embed', kernel_regularizer= l2(0.0005))(net)
    net = Dropout(0.5)(net)
    net = Dense(embedding_dim, activation='relu', name='embed2', kernel_regularizer= l2(0.0005))(net)
    net = Dropout(0.5)(net)
    net = Dense(embedding_dim, activation='relu', name='embed3', kernel_regularizer= l2(0.0005))(net)
    net = Lambda(l2Norm, output_shape=[embedding_dim])(net)
    
    base_model = Model(resnet_model.input, net, name = "res-net")
    
    #Inputs
    input_shape=(input_shape)
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_pos')
    input_negative = Input(shape=input_shape, name='input_neg')
    
    embedded_anchor = base_model(input_anchor)
    embedded_positive = base_model(input_positive)
    embedded_negative = base_model(input_negative)
    
    #Distances: Here Euclidian
    dist_AB = Lambda(euclidean_distance, name='pos_dist')([embedded_anchor, embedded_positive])
    dist_AC = Lambda(euclidean_distance, name='neg_dist')([embedded_anchor, embedded_negative])

    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1),
            name='stacked_dists')([dist_AB, dist_AC])
    
    inputs = [input_anchor, input_positive, input_negative]

    
    siamese_net = Model(inputs=inputs, outputs=stacked_dists, name='triple_siamese')
    
    # return the model
    return siamese_net

#####################################################################################################
    
#Self-written DataGenerator function to generate batches for the Keras fit function 
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, train_data, train_triplets , batch_size=32, dim=(128,128), n_channels=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        #Added
        self.train_data = train_data
        self.train_triplets = train_triplets 
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        n_row = self.train_triplets.shape[0]
        return int(np.floor(n_row / self.batch_size))
        
        
    def __getitem__(self, index): 
        'Generate one batch of data'
        # Generate indexes of the batch
        #print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs      
        list_IDs_temp = self.train_triplets.iloc[indexes,]
            
        # Generate data
        [anchor, positive, negative] = self.__data_generation(list_IDs_temp)
        y_train = np.random.randint(2, size=(1,2,self.batch_size)).T

        return [anchor,positive, negative], y_train
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        n_row = self.train_triplets.shape[0]
        self.indexes = np.arange(n_row)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # anchor positive and negatives: (n_samples, *dim, n_channels)
        # Initialization
        anchor = np.zeros((self.batch_size,*self.dim,self.n_channels))
        positive = np.zeros((self.batch_size,*self.dim,self.n_channels))
        negative = np.zeros((self.batch_size,*self.dim,self.n_channels))
            
        nrow_temp = list_IDs_temp.shape[0] 
        # Generate data
        for i in range(nrow_temp):
            list_ind = list_IDs_temp.iloc[i,]
            anchor[i] = self.train_data[list_ind[0]]
            positive[i] = self.train_data[list_ind[1]]
            negative[i] = self.train_data[list_ind[2]]
            
            #Final Data
            anchor =  np.array(anchor, dtype = "float")
            positive = np.array(positive, dtype = "float")
            negative = np.array(negative, dtype = "float")
                
            #Set dummy variable if it doesnt work
        return [anchor, positive, negative]
    
#################################################################################

#Fit the model
model = get_siamese_model((250,250,3),  embedding_dim=128)
model.summary()

optimizer = Adam(lr = 0.00005)
model.compile(loss= tripl_loss, optimizer=optimizer, metrics = [accuracy2])
 
# Parameters
params = {'dim': (250,250),
              'batch_size': 32,
              'n_channels': 3,
              'shuffle': True}


Train_Gen = DataGenerator(train_data, train_triplets, **params)


model.fit_generator(generator=Train_Gen,
                        epochs = 2,
                        use_multiprocessing=True,
                        workers = 4,
                        verbose = 1)

# save model and architecture to single file
model.save("/content/drive/My Drive/IML Project 4 Data/resnet50_250_model.h5")

##########################################################################################
#Predictions 

test_images = train_images
test_data = train_data

#Picture numbers to create test triplets
test_triplets = pd.read_csv("/content/drive/My Drive/IML Project 4 Data/test_triplets.txt", sep=" ", header=None)
n_row = test_triplets.shape[0]


#Load Model
from keras.models import load_model   
model = load_model("/content/drive/My Drive/IML Project 4 Datar/resnet50_250_model.h5", compile = False)  


#Loop to create predictions
batch_size = n_row
h=w=250

pred = np.zeros(batch_size)

for i in range(batch_size):

    if i%100 == 0: print(i)
    list_ind = test_triplets.iloc[i,]  
    anchor =  test_data[list_ind[0]]
    positive = test_data[list_ind[1]]
    negative = test_data[list_ind[2]]
    
    anchor =  np.reshape(np.array(anchor, dtype = "float"), (1 , 250, 250, 3))
    positive = np.reshape(np.array(positive, dtype = "float"), (1, 250, 250,3))
    negative =  np.reshape(np.array(negative, dtype = "float"), (1, 250, 250,3))
    
    #print("For Pred")
    final_pred = model.predict_on_batch([anchor, positive, negative]) #predict
  
    dist_AB = final_pred[:,0,:]
    dist_AC = final_pred[:,1,:]
    
    if dist_AB < dist_AC:
        pred[i] = 1
    else: 
        pred[i] = 0
 

#Save predictions
np.savetxt("resnet50_250_PREDS.txt",pred, fmt="%d")