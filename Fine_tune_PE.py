
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


# In[342]:


def dump_info(name, obj):
	print("{0} :".format(name))
	try:
		print("   .value: {0}".format(obj.value))
		for key in obj.attrs.keys():
			print("		.attrs[{0}]:  {1}".format(key, obj.attrs[key]))
	except:
		pass


# Flags 
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('SNR', None, 'SNR of training data') 
tf.app.flags.DEFINE_float('Loaded_SNR', None, 'SNR of loaded model') 
tf.app.flags.DEFINE_float('learning_rate_num', None, 'Learning rate of training') # Learning rate
tf.app.flags.DEFINE_integer('generation', None, '# of Training generation (one batch size)')
tf.app.flags.DEFINE_float('keep_prob_num', None, 'keep prob of dropout layer')

snr = FLAGS.SNR
loaded_snr = FLAGS.Loaded_SNR

data_folder = '/work1/leo830227/GW_training_data_EOBNRv2_whitened_template_HDF5_mass_labeled/data_normalized_templates_with_label_mass_123/'
h5_file_name = 'data_normalized_templates_with_label_mass.h5'
paths = '{0}/{1}'.format(data_folder, h5_file_name) #data path


# Load training data
s = timeit.default_timer()

fh = h5py.File(paths, 'r')

for key in fh.keys():
	print(key) 

fh.visititems(dump_info)

e = timeit.default_timer()

print(e - s,'sec')


new_trainX = fh['train']['trainX'].value
new_trainY = fh['train']['train_mass'].value
new_trainValX = fh['train']['trainValX'].value
new_trainValY = fh['train']['trainVal_mass'].value
new_testX = fh['train']['testX'].value
new_testY = fh['train']['test_mass'].value

fh.close()


# Add noise to data
noise_train_shape = np.shape(new_trainX)
noise_val_shape = np.shape(new_trainValX)
noise_test_shape = np.shape(new_testX)

noise_array_train = np.random.normal(0.0, 1.0, noise_train_shape)
noise_array_val = np.random.normal(0.0, 1.0, noise_val_shape)
noise_array_test = np.random.normal(0.0, 1.0, noise_test_shape)

new_trainX = snr * trainX + noise_array_train
new_trainValX = snr * trainValX + noise_array_val
new_testX = snr * testX + noise_array_test


# Check the shape of training, validation and testing set
print('new_train:\t', np.shape(new_trainX), np.shape(new_trainY))
print('new_trainVal:', np.shape(new_trainValX), np.shape(new_trainValY))
print('new_test:\t', np.shape(new_testX), np.shape(new_testY))


save_folder_name = '{0}_SNR'.format(loaded_snr)

for file in os.listdir('/home/leo830227/result_deeper/saved_model/{0}'.format(save_folder_name)):
	if file.endswith('.meta'):
		save_file_name = file

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


# Restore the graph of DNN
restorer = tf.train.import_meta_graph('/home/leo830227/result_deeper/saved_model/{0}/{1}'.format(save_folder_name, save_file_name))
# Restore the variables
restorer.restore(sess,tf.train.latest_checkpoint('/home/leo830227/result_deeper/saved_model/{0}'.format(save_folder_name)))


# Parameters of the model
batch_size = 128
learning_rate_num = FLAGS.learning_rate_num 
#learning_rate_num = 0.001
keep_prob_num = FLAGS.keep_prob_num
#keep_prob_num = 1.0
evaluation_size = 100
time_seires_length = new_trainX[0].shape[0]

target_size = 1 + 1
num_channels = 1 
generation = FLAGS.generation
eval_every = 20 


# Restore parameter and operation by name
graph = tf.get_default_graph()

x_input = graph.get_tensor_by_name("x_input:0")
y_target = graph.get_tensor_by_name("Y_target:0")
loss = graph.get_tensor_by_name('loss:0')
prediction = graph.get_tensor_by_name('fully_connected3:0')
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



writer = tf.summary.FileWriter(
	'/home/leo830227/result_deeper/tensorboard/{0}_train'.format(next_trainning_times), sess.graph)


# Creat a saver
saver=tf.train.Saver(max_to_keep=1)

# Beccause I can't load whole test et in once (it will out of memeory), so I have to redefine the 
# accuracy function
def get_num_correct_test_set(logits, targets):
	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return(num_correct)


# Creat folder for result
os.makedirs('/home/leo830227/result_deeper_dropout/saved_model/{0}_SNR'.format(snr), exist_ok=True)
os.makedirs('/home/leo830227/result_deeper_dropout/accuracy/{0}_SNR'.format(snr), exist_ok=True)
os.makedirs('/home/leo830227/result_deeper_dropout/tensorboard/{0}_SNR'.format(snr), exist_ok=True)
new_save_folder_name = '{0}_SNR'.format(snr)
new_save_file_name = '{0}_SNR_PE'.format(snr)

# Training
start_time2 = timeit.default_timer()
# Used for record the best model
min_loss = 100.
min_val_loss = 100.
report_times = 0


#for save the result
train_loss = []
val_loss = []
train_relative_loss = []
val_relative_loss = []
mass_label_save_train = []
mass_label_save_val = []
train_pred_save = []
val_pred_save = []


for i in range(generation):
	rand_index = np.random.choice(len(new_trainX), size=batch_size, replace=None)
	rand_x = new_trainX[rand_index]
	rand_x = np.expand_dims(rand_x, 2)
	rand_y = new_trainY[rand_index]
	train_dict = {x_input: rand_x, y_target: rand_y, keep_prob: keep_prob_num, learning_rate: learning_rate_num}
	sess.run(new_train_step, feed_dict=train_dict)
	temp_train_loss, temp_train_preds, temp_train_loss_tb = sess.run([loss, prediction, loss_scalar], feed_dict=train_dict)
	temp_train_acc = get_accuracy(temp_train_preds, rand_y)
	
	# For validation
	
	eval_index = np.random.choice(len(new_trainValX), size=evaluation_size)
	eval_x = new_trainValX[eval_index]
	eval_x = np.expand_dims(eval_x, 2)
	eval_y = new_trainValY[eval_index]
	test_dict = {x_input: eval_x, y_target: eval_y, keep_prob: 1.0}
	test_loss, test_preds, test_loss_tb = sess.run([loss, prediction, val_loss_scalar], feed_dict=test_dict)
	temp_test_acc = get_accuracy(test_preds, eval_y)


	# save the best model
	if temp_train_loss < min_loss:
		min_loss = temp_train_loss
		if test_loss < min_val_loss:
			min_val_loss = test_loss
			
			# save model
			saver.save(sess,'/home/leo830227/result_deeper_dropout/saved_model/{0}/{1}'.format(new_save_folder_name, new_save_file_name), global_step=i+1)
			
			
	# Record and print results
	if (i+1) % eval_every == 0:
		
		print('Generation: {0}.'.format(i+1))
		print('Train:')
		for k in range(np.shape(rand_x)[0]):
			print('Mass Label: {0}. Prediction: {1}. Relative_loss: {2}.'.format(rand_y[k,:], temp_train_preds[k,:], temp_train_acc[k]))
		print('Loss: {0}'.format(temp_train_loss))
		
		print('############################################################################################')
		
		print('Validation:')
		for m in range(np.shape(eval_x)[0]):
			print('Mass Label: {0}. Prediction: {1}. Delta Mass: {2}.'.format(eval_y[m,:], test_preds[m,:], temp_test_acc[m]))
		print('Loss: {0}'.format(test_loss))
	

		# save the training result
		train_loss.append(temp_train_loss)
		val_loss.append(test_loss)
		train_pred_save.append(temp_train_preds)
		val_pred_save.append(test_preds)
		mass_label_save_train.append(rand_y)
		mass_label_save_val.append(eval_y)
		train_relative_loss.append(temp_train_acc)
		val_relative_loss.append(temp_test_acc)
		

		#tensorboard
		summary = sess.run(merged, feed_dict = train_dict)
		writer.add_summary(summary, i)
		writer.add_summary(temp_train_loss_tb, i)
		writer.add_summary(test_loss_tb, i)
		
		report_times = report_times + 1
		
	
stop_time2 = timeit.default_timer()

training_time_cost = stop_time2 - start_time2


# reshape the result to save.
train_reshape_shape = (report_times * batch_size, -1)
val_reshape_shape = (report_times * evaluation_size, -1)
train_pred_save = list(np.reshape(np.asarray(train_pred_save), train_reshape_shape))
val_pred_save = list(np.reshape(np.asarray(val_pred_save), val_reshape_shape))
mass_label_save_train = list(np.reshape(np.asarray(mass_label_save_train), train_reshape_shape))
mass_label_save_val = list(np.reshape(np.asarray(mass_label_save_val), val_reshape_shape))
train_relative_loss = list(np.reshape(np.asarray(train_relative_loss), train_reshape_shape))
val_relative_loss = list(np.reshape(np.asarray(val_relative_loss), val_reshape_shape))


# save h5 file
gh = h5py.File('/home/leo830227/result_deeper_dropout/accuracy/{0}_SNR/result.h5'.format(snr))


gh1 = gh.create_group('Meta')
gh2 = gh.create_group('Loss')
gh3 = gh.create_group('Prediction')
gh4 = gh.create_group('Mass_label')
gh5 = gh.create_group('Relative_loss')
gh6 = gh.create_group('SNR')

gh1['Infromation'] = 'm1 and m2 parameter estimation result.'
#gh1['SNR'] = snr
gh1['Learning_rate'] = learning_rate_num
gh1['Keep_prob'] = keep_prob_num
mass_label_tmp = np.vstack((new_trainY, new_trainValY))
mass_label = list(np.vstack((mass_label_tmp, new_testY)))
gh1['Mass_label'] = mass_label
gh1['train, validation, test #'] = [np.shape(new_trainX)[0], np.shape(new_trainValX)[0], np.shape(new_testX)[0]]
gh1['Training_times'] = training_time_cost
gh1['Handsome'] = 'Leo'

gh2['Loss_function'] = 'Mean absolute relative error function, mean(abs(true answer - prediction) / true answer)'
gh2['Train_loss'] = train_loss
gh2['Validation_loss'] = val_loss

gh3['Train_prediction'] = train_pred_save
gh3['Validation_prediction'] = val_pred_save

gh4['Train_mass_label'] = mass_label_save_train
gh4['Validation_mass_label'] = mass_label_save_val

gh5['Function'] = 'abs(true answer - prediction) / true answer * 100, means percent.'
gh5['Information'] = '[m1 relative loss, m2 relative loss].'
gh5['Train_relative_loss'] = train_relative_loss
gh5['Validation_relative_loss'] = val_relative_loss
#f.close()
writer.close()


print(stop_time2 - start_time2) 


#  ----------------------------------------------------------------------------------------------------------------

## Test

tf.reset_default_graph() # Reset the variable of tensorflow


# Load the training model
for file in os.listdir('/home/leo830227/result_deeper_dropout/saved_model/{0}'.format(new_save_folder_name)):
	if file.endswith('.meta'):
		save_file_name = file

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


# Restore the graph of DNN
restorer = tf.train.import_meta_graph('/home/leo830227/result_deeper_dropout/saved_model/{0}/{1}'.format(new_save_folder_name, new_save_file_name))
# Restore the variables
restorer.restore(sess,tf.train.latest_checkpoint('/home/leo830227/result_deeper_dropout/saved_model/{0}'.format(new_save_folder_name)))


st = timeit.default_timer()

test_size = np.shape(new_testX)[0]

test_index = np.arange(test_size)
cut_number = 17

# Cut the test set into peices, avoiding out of memeory problem.
test_size_input = int(test_size / cut_number)
for i in range (0,cut_number):
	locals()['test_index_%s'%(i+1)] = test_index[test_size_input * i: test_size_input * (i + 1)]
#print(np.shape(test_index_2))
for i in range(1, cut_number + 1):
	locals()['test_x_%s'%i]= new_testX[locals()['test_index_%s'%i]]
	locals()['test_x_%s'%i] = np.expand_dims(locals()['test_x_%s'%i], 2)
	locals()['test_y_%s'%i]= new_testY[locals()['test_index_%s'%i]]	

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
############################################################################################

"""
print('Test:')
for k in range(np.shape(rand_x)[0]):
	print('Mass Label: {0}. Prediction: {1}. Relative_loss: {2}.'.format(test_mass_label_np[k,:], result_preds_np[k,:], test_relative_loss_np[k,:]))
"""

# save result into HDF5 file.
gh3['Test_prediction'] = result_preds_np_list
gh4['Test_mass_label'] = test_mass_label_np_list
gh5['Test_relative_loss'] = test_relative_loss_np_list

# for final trainig
gh6['SNR'] = SNR
gh6['Loaded_SNR'] = loaded_snr 

gh.close()

et = timeit.default_timer()

