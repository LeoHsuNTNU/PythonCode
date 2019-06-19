
# coding: utf-8

# In[341]:


import os
import pylab
import tensorflow as tf
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns
import pandas as pd
import numpy as np
import math
import timeit
from collections import OrderedDict
import h5py
import sklearn.preprocessing


# In[342]:


def dump_info(name, obj):
	print("{0} :".format(name))
	try:
		print("   .value: {0}".format(obj.value))
		for key in obj.attrs.keys():
			print("		.attrs[{0}]:  {1}".format(key, obj.attrs[key]))
	except:
		pass


# In[343]:


#CUDA_VISIBLE_DEVICES = "0,1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# In[344]:


# Flags for python file.
# for training
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('Name', None, 'Event name') # SNR of training data


"""
#for fintal training
tf.app.flags.DEFINE_string('SNR', 'final', None)
tf.app.flags.DEFINE_float('learning_rate_num', 0.001, 'Learning rate of training') # Learning rate
#tf.app.flags.DEFINE_integer('trainning_times', , 'Which time of trainning')
tf.app.flags.DEFINE_integer('generation', 800000, '# of Training generation (one batch size)')
tf.app.flags.DEFINE_float('keep_prob_num', 1.0, 'keep prob of dropout layer')
"""

event_name = FLAGS.Name

data_folder = '/home/leo830227/LOSC_Event_tutorial/LOSC_Event_tutorial'
h5_file_name = '{0}.hdf5'.format(event_name)


# In[345]:

# commen out for final training
"""
trainning_times = FLAGS.trainning_times
#trainning_times = 19
next_trainning_times = trainning_times + 1
print(next_trainning_times)
"""
#next_trainning_times = 'Trainsfer'



# ####################################################################################################

# In[346]:


paths = '{0}/{1}'.format(data_folder, h5_file_name) #data path
#print(paths)


# In[347]:


s = timeit.default_timer()

fh = h5py.File(paths, 'r')

for key in fh.keys():
	print(key) 

fh.visititems(dump_info)

e = timeit.default_timer()

print(e - s,'sec')


# In[348]:

new_testX = fh['Strain']['H'].value
#test_time = fh['Data']['Time'].value
new_testY = np.zeros(np.shape(new_testX))

fh.close()

# scale the data
new_testX = sklearn.preprocessing.scale(new_testX, axis=0)

# Check the shape of training, validation and testing set
print('new_test:\t', np.shape(new_testX))
# ################################################################################

# In[350]:


save_folder_name = 'final_train_20W'

for file in os.listdir('/home/leo830227/result_deeper/saved_model/{0}'.format(save_folder_name)):
	if file.endswith('.meta'):
		save_file_name = file


# 定義TensorFlow配置
config = tf.ConfigProto()

# 配置GPU內存分配方式，按需增長，很關鍵
config.gpu_options.allow_growth = True

# Start a graph session
sess = tf.Session(config = config)

# Restore the weight and model

# Restore the graph of DNN
restorer = tf.train.import_meta_graph('/home/leo830227/result_deeper_dropout/saved_model/{0}/{1}'.format(save_folder_name, save_file_name))
# Restore the variables
restorer.restore(sess,tf.train.latest_checkpoint('/home/leo830227/result_deeper_dropout/saved_model/{0}'.format(save_folder_name)))


# In[351]:


# Parameters of the model
time_seires_length = new_trainX[0].shape[0]

#############################################
# Learning rate and batch size有許多不同的取法，不完全都是固定的，有很多取法是可以隨著training變動的，ex. minibatch 
# approximation
#############################################
target_size = 1 + 1
print(target_size)
num_channels = 1 # only one channel: strain
#generation = 10000
eval_every = 20 #幾個generation回報一次
#print(generation)


# In[352]:


# parameter for new_saver
#new_SNR = round(SNR, 1)
#print(new_SNR)


"""
# for final training
new_save_folder_name = 'final_train'
new_save_file_name = 'final_train_{0}'.format(generation)
"""

# In[353]:


"""
global_steps = tf.Variable(0, name='global_steps', trainable=False)
init_new_vars_op = tf.initialize_variables([global_steps])
sess.run(init_new_vars_op)
"""
# Restore parameter and operation by name
graph = tf.get_default_graph()

x_input = graph.get_tensor_by_name("x_input:0")
#print(x_input.name)
y_target = graph.get_tensor_by_name("Y_target:0")
#eval_input = graph.get_tensor_by_name("eval_input:0")
#eval_target = graph.get_tensor_by_name("eval_target:0")
loss = graph.get_tensor_by_name('loss:0')
#val_loss = graph.get_tensor_by_name('val_loss:0')
prediction = graph.get_tensor_by_name('fully_connected3:0')
#val_prediction = graph.get_tensor_by_name('val_prediction:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

# Weight
conv1_weight = graph.get_tensor_by_name('conv1_weight:0')
conv2_weight = graph.get_tensor_by_name('conv2_weight:0')
conv3_weight = graph.get_tensor_by_name('conv3_weight:0')
conv4_weight = graph.get_tensor_by_name('conv4_weight:0')

# Learning_rate
learning_rate = graph.get_tensor_by_name('learning_rate:0')

# For tensorboard
train_accuracy_tb = graph.get_tensor_by_name('train_accuracy_tb:0')
val_accuracy_tb =  graph.get_tensor_by_name('val_accuracy_tb:0')

# Operations
new_train_step = tf.get_collection('train_step')[0]
loss_scalar = tf.get_collection('loss_scalar')[0]
val_loss_scalar = tf.get_collection('val_loss_scalar')[0]
merged = tf.get_collection('merged')[0]
train_accuracy_tb_scalar = tf.get_collection('train_accuracy_tb_scalar')[0]
val_accuracy_tb_scalar = tf.get_collection('val_accuracy_tb_scalar')[0]


# In[354]:


# Create accuracy function
def get_accuracy(logits, targets):
	
	difference = []
	batch_predictions = np.asarray(logits)
	target_np = np.asarray(targets)
	for l in range(np.shape(logits)[0]):
		difference_tmp = (np.abs(logits[l] - targets[l]) / targets[l]) * 100
		difference.append(difference_tmp)
	difference = np.asarray(difference)
	return(difference)


# Creat a saver
saver=tf.train.Saver(max_to_keep=1)

# Beccause I can't load whole test et in once (it will out of memeory), so I have to redefine the 
# accuracy function

def get_num_correct_test_set(logits, targets):
	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return(num_correct)

def get_tp_fn_fp_and_tn(logits, targets):
	true_positives = []
	false_negatives = []
	false_positives = []
	true_negatives = []
	batch_predictions = np.argmax(logits, axis=1)
	for i in range(len(batch_predictions)):
		if batch_predictions[i] == 1:
			true_positives_tmp =  np.sum(np.equal(batch_predictions[i], targets[i]))
			true_positives.append(true_positives_tmp)
		if batch_predictions[i] == 0:
			false_negatives_tmp = np.sum(np.not_equal(batch_predictions[i], targets[i]))
			false_negatives.append(false_negatives_tmp)
			#print('FN: ', logits[i])
		if batch_predictions[i] == 1:
			false_positives_tmp = np.sum(np.not_equal(batch_predictions[i], targets[i]))
			false_positives.append(false_positives_tmp)
			#print('FP: ', logits[i])
		if batch_predictions[i] == 0:
			true_negatives_tmp = np.sum(np.equal(batch_predictions[i], targets[i]))
			true_negatives.append(true_negatives_tmp)
	return(true_positives, false_negatives, false_positives, true_negatives)
# In[357]:

#for save the result
#os.makedirs('/home/leo830227/result_deeper_dropout/accuracy_SNR/{0}_SNR'.format(SNR),exist_ok=True)
#gh = h5py.File('/home/leo830227/result_deeper_dropout/accuracy_SNR/{0}_SNR/result.h5'.format(SNR), 'w')

start_time3 = timeit.default_timer()
#f=open('/home/leo/Leo/Deep/{0}_{1}/accuracy/{2}/{3}_test.txt'.format(apporximation, optimizer, new_save_folder_name, new_save_file_name),'w')
test_size = np.shape(new_testX)[0]

print(test_size)

cut_number = 1

test_index = np.arange(0, test_size, 1)


# Cut the test set into peices, avoiding out of memeory problem.
test_size_input = int(test_size / cut_number)
for i in range (0,cut_number):
	locals()['test_index_%s'%(i+1)] = test_index[test_size_input * i: test_size_input * (i + 1)]
#print(np.shape(test_index_2))
for i in range(1, cut_number + 1):
	locals()['test_x_%s'%i]= new_testX[locals()['test_index_%s'%i]]
	locals()['test_x_%s'%i] = np.expand_dims(locals()['test_x_%s'%i], 2)
	locals()['test_y_%s'%i]= new_testY[locals()['test_index_%s'%i]]	
	
	# for final trainig
	#locals()['test_z_%s'%i]= new_test_SNR[locals()['test_index_%s'%i]]


#print(np.shape(test_x_2))
#print(np.shape(test_y_2))
test_set_dict_x = OrderedDict([])
test_set_dict_y = OrderedDict([])
for i in range(1, cut_number + 1):
	tmp_test_set_dict_x = OrderedDict([('test_x_{0}'.format(i),locals()['test_x_%s'%i])])
	test_set_dict_x.update(tmp_test_set_dict_x)
	tmp_test_set_dict_y = OrderedDict([('test_y_{0}'.format(i),locals()['test_y_%s'%i])])
	test_set_dict_y.update(tmp_test_set_dict_y)

test_mass_label = []
result_preds = []
test_relative_loss = []


# Testing
for (k1, v1),(k2, v2) in zip(test_set_dict_x.items(), test_set_dict_y.items()):
	result_dict = {x_input: v1, y_target: v2, keep_prob: 1}
	tmp_result_preds = sess.run(prediction, feed_dict=result_dict)
	tmp_test_relative_loss = get_accuracy(tmp_result_preds, v2)
	
	result_preds.append(tmp_result_preds)
	test_mass_label.append(v2)
	test_relative_loss.append(tmp_test_relative_loss)

# reshape the result for printing out
test_mass_label_np = np.reshape(np.asarray(test_mass_label), (test_size, -1))
result_preds_np = np.reshape(np.asarray(result_preds), (test_size, -1))
test_relative_loss_np = np.reshape(np.asarray(test_relative_loss), (test_size, -1))

# to save as HDF5 file, need list, not numpy array
test_mass_label_np_list = list(test_mass_label_np)
result_preds_np_list = list(result_preds_np)
test_relative_loss_np_list = list(test_relative_loss_np)
print(result_preds_np_list)
############################################################################################

"""
print('Test:')
for k in range(np.shape(rand_x)[0]):
	print('Mass Label: {0}. Prediction: {1}. Relative_loss: {2}.'.format(test_mass_label_np[k,:], result_preds_np[k,:], test_relative_loss_np[k,:]))
"""
"""
# save result into HDF5 file.
gh.create_dataset('Test/Test_prediction', data=result_preds_np_list)
gh.create_dataset('Test/Test_mass_label', data=test_mass_label_np_list)
gh.create_dataset('Test/Test_relative_loss', data=test_relative_loss_np_list)

# for final trainig
gh.create_dataset('Test/SNR' ,data=SNR)

gh.close()


mean_relative_error = np.sum(test_relative_loss_np_list) / (np.shape(test_relative_loss_np_list)[0] * np.shape(test_relative_loss_np_list)[1])

acc_array = [[SNR, mean_relative_error]]

ga = open('/home/leo830227/result_deeper_dropout/result.txt', 'ab')

np.savetxt(ga,acc_array, fmt='%.5e') 
ga.close()

et = timeit.default_timer()
"""
