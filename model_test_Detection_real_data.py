
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

tf.app.flags.DEFINE_string('Name', None, 'Event name') 

event_name = FLAGS.Name

data_folder = '/home/leo830227/LOSC_Event_tutorial/LOSC_Event_tutorial'
h5_file_name = '{0}.hdf5'.format(event_name)


# In[345]:


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
new_testX = new_testX[1:31]
for i in range(30):
	new_testX[i] = new_testX[i] / max(new_testX[i])
new_testY = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
print(np.shape(new_testY))
#test_time = fh['Data']['Time'].value

fh.close()



"""
# Check the shape of training, validation and testing set
print('new_train:\t', np.shape(new_trainX), np.shape(new_trainY))
print('new_trainVal:', np.shape(new_trainValX), np.shape(new_trainValY))
print('new_test:\t', np.shape(new_testX), np.shape(new_testY))
"""
# ################################################################################

# In[350]:


save_folder_name = 'Trainsfer_train_bad'

for file in os.listdir('/home/leo830227/result_deeper_detection/saved_model/{0}'.format(save_folder_name)):
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
restorer = tf.train.import_meta_graph('/home/leo830227/result_deeper_detection/saved_model/{0}/{1}'.format(save_folder_name, save_file_name))
# Restore the variables
restorer.restore(sess,tf.train.latest_checkpoint('/home/leo830227/result_deeper_detection/saved_model/{0}'.format(save_folder_name)))


# In[351]:


# Parameters of the model
time_seires_length = new_testX[0].shape[0]

#############################################
# Learning rate and batch size有許多不同的取法，不完全都是固定的，有很多取法是可以隨著training變動的，ex. minibatch 
# approximation
#############################################
target_size = 1 + 1
print(target_size)
num_channels = 1 # only one channel: strain

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
	"""
	difference = []
	batch_predictions = np.asarray(logits)
	target_np = np.asarray(targets)
	for l in range(np.shape(logits)[0]):
		difference_tmp = (np.abs(logits[l] - targets[l]) / targets[l]) * 100
		difference.append(difference_tmp)
	difference = np.asarray(difference)
	return(difference)
	"""

	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return(100. * num_correct/batch_predictions.shape[0])



# Creat a saver
saver=tf.train.Saver(max_to_keep=1)

# Beccause I can't load whole test et in once (it will out of memeory), so I have to redefine the 
# accuracy function
def get_num_correct_test_set(logits):
	batch_predictions = np.argmax(logits, axis=1)
	return(batch_predictions)

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
os.makedirs('/home/leo830227/result_LIGO/{0}'.format(event_name),exist_ok=True)
gh = h5py.File('/home/leo830227/result_LIGO/{0}/{1}.h5'.format(event_name, event_name), 'w')

start_time3 = timeit.default_timer()
#f=open('/home/leo/Leo/Deep/{0}_{1}/accuracy/{2}/{3}_test.txt'.format(apporximation, optimizer, new_save_folder_name, new_save_file_name),'w')
test_size = np.shape(new_testX)[0]

print(test_size)

cut_number = 1


test_size = np.shape(new_testX)[0]

#test_index = np.random.choice(len(new_testX), size=test_size, replace=None)
test_index = np.arange(test_size)

# Cut the test set into 36 peices, avoiding out of memeory problem.
test_size_input = int(test_size / cut_number)

for i in range (0,cut_number):
	locals()['test_index_%s'%(i+1)] = test_index[test_size_input * i: test_size_input * (i + 1)]
#print(np.shape(test_index_2))
for i in range(1, cut_number + 1):
	locals()['test_x_%s'%i] = new_testX[locals()['test_index_%s'%i]]
	locals()['test_x_%s'%i] = np.expand_dims(locals()['test_x_%s'%i], 2)
	locals()['test_y_%s'%i] = new_testY[locals()['test_index_%s'%i]]
#print(test_x_3)
#print(test_y_3)
#print(test_z_32)

test_set_dict_x = OrderedDict([])
test_set_dict_y = OrderedDict([])
for i in range(1, cut_number + 1):
	tmp_test_set_dict_x = OrderedDict([('test_x_{0}'.format(i),locals()['test_x_%s'%i])])
	test_set_dict_x.update(tmp_test_set_dict_x)
	tmp_test_set_dict_y = OrderedDict([('test_y_{0}'.format(i),locals()['test_y_%s'%i])])
	test_set_dict_y.update(tmp_test_set_dict_y)
	
result_list = []
result_preds_list = []

for (k1, v1), (k2, v2) in zip(test_set_dict_x.items(), test_set_dict_y.items()):
	#result_dict = {x_input: v1, keep_prob: 1}
	result_dict = {x_input: v1, y_target: v2, keep_prob: 1}
	result_preds = sess.run(prediction, feed_dict=result_dict)
	result_preds_list.append(result_preds)
	result = get_num_correct_test_set(result_preds)

# Result of test
"""
Model_correct_num = np.sum(result_list)
Model_accuracy = 100. * Model_correct_num/test_size
TP_num = []
FN_num = []
FP_num = []
TN_num = []
for i in range(cut_number):
	TP_num_tmp = np.sum(TP[i])
	FN_num_tmp = np.sum(FN[i])
	FP_num_tmp = np.sum(FP[i])
	TN_num_tmp = np.sum(TN[i])
	TP_num.append(TP_num_tmp)
	FN_num.append(FN_num_tmp)
	FP_num.append(FP_num_tmp)
	TN_num.append(TN_num_tmp)

TP_num = np.sum(np.sum(TP_num))
FN_num = np.sum(np.sum(FN_num))
FP_num = np.sum(np.sum(FP_num))
TN_num = np.sum(np.sum(TN_num))
"""

#print(type(FN_mass))
#print(FN_mass[6])
#print(np.asarray(FN_mass))
#addasdsa


#Model_sensitivity = TP_num / (TP_num + FN_num) * 100
#print(result_preds_list)

# Print out the result.
#print('Model accuracy: {0}'.format(Model_accuracy))
#print('Model sensitivity: {0}'.format(Model_sensitivity))
#print('Confusion matrix:')
#print('{0}	   {1}'.format(TP_num, FN_num))
#print('{0}	   {1}'.format(FP_num, TN_num))
#print('{0} / {1}'.format(Model_correct_num, test_size))
#saver.save(sess,'/home/leo/Leo/Deep/{0}/saved_model/{1}/{2}'.format(apporximation, new_save_folder_name, new_save_file_name), global_step=i+1)
#FN_mass = list(np.asarray(FN_mass))
print(result_preds_list)
print(result)
# Write the result as a HDF5 file

#gh.create_dataset('/Test/Test_accuracy', data=Model_accuracy)
#gh.create_dataset('/Test/Test_sensitivity', data=Model_sensitivity)
#gh.create_dataset('/Test/TP', data=TP_num)
#gh.create_dataset('/Test/FN', data=FN_num)
#gh.create_dataset('/Test/FP', data=FP_num)
#gh.create_dataset('/Test/TN', data=TN_num)
gh.create_dataset('/Test/Prediction', data=result_preds_list)

"""
gh3['Test_accuracy'] = Model_accuracy
gh3['Test_sensitivity']  = Model_sensitivity
gh3['TP'] = TP_num
gh3['FN'] = FN_num
gh3['FP'] = FP_num
gh3['TN'] = TN_num
#gh3['FP_mass'] = FP_mass
gh3['FN_mass'] = FN_mass
"""
"""
f.write('Acc: {0},	Sensitivity: {1} (TP / (TP + FN))'.format(Model_accuracy, Model_sensitivity)+'\n')
f.write('Confusion matrix:'+'\n')
f.write('{0}	 {1}'.format(TP_num, FN_num)+'\n')
f.write('{0}	 {1}'.format(FP_num, TN_num)+'\n')
f.write('test set mumber: {0}'.format(test_size))
f.close()
"""
#sess.close()
gh.close()
"""
acc_folder = open('/home/leo830227/result_deeper_detection/result.txt','ab')

save_array = [[SNR, Model_accuracy, Model_sensitivity]]

np.savetxt(acc_folder, save_array, fmt='%.5e')

acc_folder.close()
"""


"""
test_index = np.arange(0, test_size, 1)

print(max(test_index))

# Cut the test set into peices, avoiding out of memeory problem.
test_size_input = int(test_size / cut_number)
for i in range (0, cut_number):
	locals()['test_index_%s'%(i+1)] = test_index[test_size_input * i: test_size_input * (i + 1)]
#print(np.shape(test_index_2))
for i in range(1, cut_number + 1):
	locals()['test_x_%s'%i]= new_testX[locals()['test_index_%s'%i]]
	locals()['test_x_%s'%i] = np.expand_dims(locals()['test_x_%s'%i], 2)
	locals()['test_y_%s'%i]= new_testY[locals()['test_index_%s'%i]]
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
test_acc = []

# Testing
for (k1, v1),(k2, v2) in zip(test_set_dict_x.items(), test_set_dict_y.items()):
	result_dict = {x_input: v1, y_target: v2, keep_prob: 1}
	tmp_result_preds = sess.run(prediction, feed_dict=result_dict)
	tmp_test_acc = get_accuracy(tmp_result_preds, v2)
	
	result_preds.append(tmp_result_preds)
	test_mass_label.append(v2)
	test_acc.append(tmp_test_acc)

stop_time3 = timeit.default_timer()

# reshape the result for printing out
test_mass_label_np = np.reshape(np.asarray(test_mass_label), (test_size, -1))
result_preds_np = np.reshape(np.asarray(result_preds), (test_size, -1))
#test_acc_np = np.reshape(np.asarray(test_acc), (test_size, -1))

# to save as HDF5 file, need list, not numpy array
test_mass_label_np_list = testZ
result_preds_np_list = list(result_preds_np)
#test_acc_np_list = list(test_acc_np)

test_acc = np.sum(test_acc) / np.shape(test_acc)[0]
############################################################################################


print('Test:')
for k in range(np.shape(rand_x)[0]):
	print('Mass Label: {0}. Prediction: {1}. acc: {2}.'.format(test_mass_label_np[k,:], result_preds_np[k,:], test_acc_np[k,:]))

# save result into HDF5 file.
cost_time2 = stop_time3 - start_time3

gh['Meta/Test_times'] = cost_time2
gh['Prediction/Test_prediction'] = result_preds_np_list
gh['Mass_label/Test_mass_label'] = test_mass_label_np_list
gh['Result/Test_acc'] = test_acc

gh.close()

print(test_acc)
"""
