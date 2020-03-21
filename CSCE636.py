#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Used for Part 1:
import numpy as np
import h5py
import random
# Used for Part 2:
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
# Used for Part3:
from keras.layers import Input, Dense, Permute, Reshape
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Add
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
import matplotlib.pylab as pl
import tensorflow as tf
#Used for Part4:
from keras.models import load_model
from keras_bert import get_custom_objects
import json


# In[66]:


# Part 1: Generate Data
class Generate_Data_Input(object):
    
    def __init__(self, path, batches = 64, frames = 30):
        self.path = path
        self.batches = batches
        self.frames = frames
        
    def Data_Name(self):
        
        #Read all names of video datasets and return a list
        
        with h5py.File(self.path, 'r') as read_file:
            return [data_name for data_name in read_file]
    
    
    def Cursors(self, data_name):
        
        # The input is the name of the video
        # In a 3 seconds video, it is nearly 100 cursors, the function 'cursors' is in order 
        # to output a 1 * 30 random array to represnt the cursor index we will extract from a video
        
        with h5py.File(self.path, 'r') as read_file:
            data = read_file[data_name]
            data_frames = data.shape[1]
        
        output_cursors = []
        extra_frames = data_frames % self.frames
        group_num = data_frames // self.frames
        
        max_number = group_num * (self.frames +  1) - 1
        min_number = group_num * 1 - 1
        
        cursors_index = np.arange(min_number, max_number, group_num)
        
        if extra_frames != 0:
            extra_cursors = sorted(random.sample(range(self.frames), extra_frames))
            
            for index in extra_cursors:
                cursors_index[index:] = cursors_index[index:] + 1
        
        output_cursors.append(random.randint(0, cursors_index[0]))
        
        for index2 in range(1, self.frames):
            output_cursors.append(random.randint(cursors_index[index2 - 1] + 1, cursors_index[index2]))
        #output_cursors = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
        #40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] JSON File Label.
            
        return output_cursors

    
    def Train_Data(self, data_name, output_cursors, skeleton_index):
        
        # Generate the format of train data, also generate the cursor difference and the label
        
        initial = np.zeros([1, 25,3], dtype = float)
        sequence = [23, 24, 11, 10, 9, 8, 4, 5, 6, 7, 22, 21, 19, 18, 17, 16, 12, 13, 14, 15, 3, 2, 20, 1, 0]
        with h5py.File(self.path,'r') as read_file:
            single_train_data = read_file[data_name]
            output = single_train_data[skeleton_index, output_cursors, :, :]
            output = output[:, sequence ,:]
            output = np.array(output, dtype='float32')
            cursor_diff = np.diff(output, n=1, axis=0)
            cursor_diff = np.array(np.append(cursor_diff, initial, axis=0), dtype='float32')
            output = np.expand_dims(output, axis=0)
            cursor_diff = np.expand_dims(cursor_diff, axis=0)
            label = np.array(single_train_data.attrs['label']).reshape((60,))
        return output, cursor_diff, label
    
    
    def Test_Data(self, data_name, output_cursors, skeleton_index):
        
        # Generate the format of test data, also generate the cursor difference and the label
        
        initial = np.zeros([1, 25, 3], dtype = float)
        sequence = [23, 24, 11, 10, 9, 8, 4, 5, 6, 7, 22, 21, 19, 18, 17, 16, 12, 13, 14, 15, 3, 2, 20, 1, 0]
        with h5py.File(self.path,'r') as read_file:
            single_test_data = read_file[data_name]
            output = single_test_data[skeleton_index, output_cursors, :, :]
            output = output[:, sequence, :]
            output = np.array(output, dtype='float32')
            cursor_diff = np.diff(output, n=1, axis=0)
            cursor_diff = np.array(np.append(cursor_diff, initial, axis=0), dtype='float32')
            output = np.expand_dims(output, axis=0)
            cursor_diff = np.expand_dims(cursor_diff, axis=0)
            label = np.array(single_test_data.attrs['label']).T 
        return output, cursor_diff, label
            
    
    def Batch_Generator(self, total_num): 
        
        #Generate batch index based on the total number of dataset
        
        b_out = []
        extra = total_num % self.batches
        group = total_num // self.batches
        cursors = np.arange(self.batches, total_num, self.batches)
        b_out.append(list(range(0, cursors[0])))
        
        for i in range(1, group):
            b_out.append(list(range(cursors[i -1], cursors[i])))
        
        if extra != 0:
            number = self.batches - extra
            c_num = list(random.sample(range(0, total_num), number))
            output_cursor = list(range(cursors[-1], total_num))
            output_cursors = output_cursor + c_num
            b_out.append(output_cursors)
        return b_out
        
    def Data_Batch_Generator(self, name_list, single_batch, skeleton_index):
        
        # Generate batch data's matrix, difference and labels
        
        data_batch = []
        diff_d_batch = []
        labels = []
        for i in single_batch:
            data_name = name_list[i]
            output_cursors = self.Cursors(data_name = data_name)
            data_output, diff_data_output, label = self.Train_Data(data_name=data_name, output_cursors=output_cursors, 
                                                                   skeleton_index=skeleton_index)
            data_batch.append(data_output)
            labels.append(label)
            diff_d_batch.append(diff_data_output)
        
        data_batch = np.array(data_batch, dtype='float32').reshape((self.batches, self.frames, 25, 3))
        diff_d_batch = np.array(diff_d_batch, dtype='float32').reshape((self.batches, self.frames, 25, 3))
        labels = np.array(labels, dtype='float32')
        
        return data_batch, diff_d_batch, labels
    


# In[67]:


if __name__ == '__main__':
    path = 'D:/ChromeDownload/cross_subject/Test.hdf5'
    data = Generate_Data_Input(path, 32, 30)
    
    #Test here:
    #namelist = data.Data_Name()
    #print(namelist[537])
    cursors = data.Cursors(namelist[0])
    print(cursors)
    #train_data, diff, label = data.Test_Data(namelist[0], cursors, 0)
    #print(train_data)
    #s = np.array([[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]])
    #print(np.shape(s))
    
    #print(len(namelist))
    #k = data.Batch_Generator(len(namelist))
    #print(k)
    #n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    #data_batch, diff_d_batch, labels = data.Data_Batch_Generator(namelist,n , 0)
    #print(len(data_batch[0]))


# In[68]:


# Part 2: Add Custom Layer: Transformer
class Transformer(Layer):
    
    def __init__(self, d_k = 30, frames = 30, **kwargs):
        self.d_k = d_k
        self.frames = frames
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='transformer',
                                 shape=[input_shape[2], self.d_k],
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):

        inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))
        inputs = K.reshape(inputs, (-1, self.frames*3, int(inputs.shape[3])))
        inputs = K.dot(inputs, self.W)
        inputs = K.reshape(inputs, (-1, 3, self.frames, self.d_k))
        inputs = K.permute_dimensions(inputs, (0, 2, 3, 1))
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_k, input_shape[3])


# In[69]:


#Part 3: Train model

# Define Hyper-Parameter:
epochs = 200
batch_size = 64
frames = 30
learning_rate = 0.001 #change from 0.001
d_k = 30
input_shape = (frames, 25, 3)
use_bias = True

# File Load and Save Path:
train_path = 'D:/ChromeDownload/cross_subject/Train.hdf5'
tst_path = 'D:/ChromeDownload/cross_subject/Test.hdf5'
weight_path = 'D:/636/add_weights.h5'
model_path = 'D:/636/add_model.h5'
graph_path = 'D:/636/model_v1.png'


def Share_Stream(x_shape):
    
    # Share Layers in Two-stream CNN
    
    x = Input(shape=x_shape)
    x_a = Transformer(d_k, frames)(x)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(x_a)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)

    shared_layer = Model(x, conv3)
    return shared_layer


def Model_Design():
    
    # Two-strem CNN Model Design
    
    up_0 = Input(shape=input_shape, name='up_stream_0')
    up_1 = Input(shape=input_shape, name='up_stream_1')
    down_0 = Input(shape=input_shape, name='down_stream_0')
    down_1 = Input(shape=input_shape, name='down_stream_1')

    up_stream = Share_Stream(x_shape=input_shape)
    down_stream = Share_Stream(x_shape=input_shape)

    up_feature_0 = up_stream(up_0)
    up_feature_1 = up_stream(up_1)
    down_feature_0 = down_stream(down_0)
    down_feature_1 = down_stream(down_1)

    up_feature_0 = Flatten()(up_feature_0)
    up_feature_1 = Flatten()(up_feature_1)
    down_feature_0 = Flatten()(down_feature_0)
    down_feature_1 = Flatten()(down_feature_1)

    up_feature = Add()([up_feature_0, up_feature_1]) # Change Here: Add, Maximum, Average, Multiply
    down_feature = Add()([down_feature_0, down_feature_1])

    feature = concatenate([up_feature, down_feature])

    fc_1 = Dense(units=256, activation='relu', kernel_regularizer=l2(0.001))(feature)
    fc_1 = Dropout(0.5)(fc_1)

    fc_2 = Dense(units=60, activation='softmax')(fc_1)

    network = Model(input=[up_0, up_1, down_0, down_1], outputs=fc_2)
    return network

def Train_Model(network):
    
    #Train Model
    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    network.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    network.summary()
    plot_model(network, to_file=graph_path)

    batch_num = 0
    model_save_acc = 0
    all_train_accuracy = []
    all_train_loss = []
    all_tst_accuracy = []

    tst_data = Generate_Data_Input(tst_path, batch_size, frames)
    tst_data_name = tst_data.Data_Name() 

    tst_cursors = [tst_data.Cursors(name) for name in tst_data_name]
    
    for epoch in range(epochs): # For 0 to 200
        accuracy_list = []
        loss_list = []
        print(epoch + 1, ' epoch is beginning......')
        train_data = Generate_Data_Input(train_path, batch_size, frames)
        train_data_name = train_data.Data_Name()
        
        train_data_cursors = train_data.Batch_Generator(len(train_data_name))  #train data number = 39780， test data number = 16324
        index_num = random.sample(range(len(train_data_cursors)), len(train_data_cursors))
        
        for ind in index_num:
            batch_num += 1
            up_data_0, down_data_0, train_labels_0                 = train_data.Data_Batch_Generator(train_data_name, train_data_cursors[ind], 0)
            up_data_1, down_data_1, train_labels_1                 = train_data.Data_Batch_Generator(train_data_name, train_data_cursors[ind], 1)
            train_loss = network.train_on_batch([up_data_0, up_data_1, down_data_0, down_data_1], train_labels_0)
            
            #Train_Loss[1]: Loss, Train_Loss[0]: Accuracy
            accuracy_list.append(train_loss[1])
            loss_list.append(train_loss[0])
            
            if batch_num % 50 == 0:
                print('the %r batch: loss: %r  accuracy: %r' % (batch_num, train_loss[0], train_loss[1]))
        
        # Calculate the total train accuracy and loss: 
        epoch_accuracy = sum(accuracy_list) / len(accuracy_list)
        epoch_loss = sum(loss_list) / len(loss_list)
        all_train_accuracy.append(epoch_accuracy)
        all_train_loss.append(epoch_loss)
        print('the %r epoch: mean loss: %r    mean accuracy: %r' % (epoch + 1, epoch_loss, epoch_accuracy))
    
    
        # Total Test Accuracy:
        if epoch >= 0:
            tst_accuracy_list = []
            for num in range(len(tst_data_name)):
                tst_up_0, tst_down_0, tst_labels_0 =                     tst_data.Test_Data(tst_data_name[num], tst_cursors[num], 0)
                tst_up_1, tst_down_1, tst_labels_1 =                     tst_data.Test_Data(tst_data_name[num], tst_cursors[num], 1)
                tst_loss = network.test_on_batch([tst_up_0, tst_up_1, tst_down_0, tst_down_1], tst_labels_0)
                #print(tst_loss)
                tst_accuracy_list.append(tst_loss[1])
            tst_accuracy = sum(tst_accuracy_list) / len(tst_accuracy_list)
            all_tst_accuracy.append(tst_accuracy)
            print('The test data accuracy: %r' % tst_accuracy)
            if tst_accuracy > model_save_acc:
                network.save(model_path)
                network.save_weights(weight_path)
                model_save_acc = tst_accuracy
                
                
    # Plot Train(Test)'s Accuracy and Loss          
    pl.figure()
    trn_acc = pl.subplot(2, 2, 1)
    trn_loss = pl.subplot(2, 2, 2)
    tst_acc = pl.subplot(2, 1, 2)

    pl.sca(trn_acc)
    pl.plot(range(len(all_train_accuracy)), all_train_accuracy, label='Train Accuracy')
    pl.xlabel('Epoch')
    pl.ylabel('Accuracy')
    pl.ylim(0, 1.0)

    pl.sca(trn_loss)
    pl.plot(range(len(all_train_loss)), all_train_loss, label='Loss')
    pl.xlabel('Epoch')
    pl.ylabel('Loss')
    pl.ylim(0, 5.0)

    pl.sca(tst_acc)
    pl.plot(range(len(all_tst_accuracy)), all_tst_accuracy, label='Test Accuracy')
    pl.xlabel('Epoch')
    pl.ylabel('Accuracy')
    pl.ylim(0, 1.0)

    pl.legend()
    pl.show()


# In[70]:


# Main Function:

#if __name__ == '__main__':
    #network = Model_Design()
    #Train_Model(network)


# In[78]:


# Part4: Test Final Model:
# Having put all tst data in a hdf5 file, you can use the test video from h5 file directly by changeing the index.

path = 'D:/ChromeDownload/cross_subject/Test.hdf5'
Model_Path = 'D:/ChromeDownload/cross_subject/Final/add_model.h5'

data = Generate_Data_Input(path, 32, 30)
# Define Custom Layer using Keras_bert:
custom_objects = get_custom_objects()
my_objects = {'Transformer': Transformer}
custom_objects.update(my_objects)
model = load_model(Model_Path, custom_objects = custom_objects)




#----Changing Index Here to see whether the action is brush teeth-----

test_data_number = int(input("Input The Video Index:")) - 1 # Test Here, brush teeth: 2, 61, 180, not brush teeth: 4, 200， you can input every number, there are 16324 test videos.
print("The Video Name is:", namelist[test_data_number])
#---------------------------------------------------------------------



output = data.Cursors(namelist[test_data_number])

data = Generate_Data_Input(path, 32, 30)
namelist = data.Data_Name()

#---------- Code here is for generating JSON file----------------

#with h5py.File(path, 'r') as read_file:
            #data1 = read_file[namelist[test_data_number]]
            #data1_frames = data1.shape[1]
#print(data1_frames)
#range_num = data1_frames - 30 

#Json Label:
#Start Here:
#output = np.zeros((range_num, 30))
#print(np.shape(output))

#for i in range(range_num):
    #for j in range(30):
        #output[i][j] = i + j

#print(np.shape(output))
#print(output[0])

#json_label = np.zeros((range_num,1))


#for i in range(range_num):
    #outputdata_0, diffoutput_0, label_0 = data.Test_Data(namelist[test_data_number], output[i], 0)
    #outputdata_1, diffoutput_1, label_1 = data.Test_Data(namelist[test_data_number], output[i], 1)
    
    #a = (outputdata_0, outputdata_1, diffoutput_0, diffoutput_1)
    #m = list(a)
    #predict_label = model.predict(m)
    #json_label[i][0]= predict_label[0][2]
    #print(predict_label[0][2])
#print(json_label)
#End Here

#Time Function:
#output_json_file = np.zeros((range_num,2))
#t = 3
#per_cus = 3 / data1_frames
#for i in range(range_num):
    #output_json_file[i][0] = per_cus * 30 + (i + 1) * per_cus
    #output_json_file[i][1] = json_label[i][0]
#print(output_json_file)

#output_json_file=['Brush Teeth:',output_json_file.tolist()]
#json_path = 'D:/636/S001C002P006R002A003.json'
#with open(json_path,'w') as f:
        #json.dump(output_json_file, f)


#pl.plot(output_json_file, json_label)
#pl.xlabel('Time')
#pl.ylabel('Label')
#pl.ylim(0, 1.0)
# Time Fucntion End Here.
#-----------------------------------------------------------------


outputdata_0, diffoutput_0, label_0 = data.Test_Data(namelist[test_data_number], output, 0)
outputdata_1, diffoutput_1, label_1 = data.Test_Data(namelist[test_data_number], output, 1)
#print(np.shape(outputdata_0))

a = (outputdata_0, outputdata_1, diffoutput_0, diffoutput_1)
m = list(a)
predicted = model.predict(m) #Output a 1*60 Matrix which contains probability for 60 actions
#print(predicted)

yhat = np.max(predicted)
print("The probability of brush teeth is:", predicted[0][2])

for action_index in range(60):
    if predicted[0][action_index] == yhat:
        if action_index == 2:
            print("The Action in the Video is Brush Teeth")
        else:
            print("The Action in the Video is Not Brush Teeth")


# In[ ]:




