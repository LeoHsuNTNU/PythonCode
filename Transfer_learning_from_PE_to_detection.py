

# coding: utf-8

# # Part I: 資料預處理

# In[1]:


import os
import pylab
import tensorflow as tf
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib notebook')
import seaborn as sns
import pandas as pd
import numpy as np
import math
import timeit
from collections import OrderedDict
import sklearn.preprocessing
import h5py


# ### 抓取input data的path

# In[2]:


def dump_info(name, obj):
	print("{0} :".format(name))
	try:
		print("   .value: {0}".format(obj.value))
		for key in obj.attrs.keys():
			print("		.attrs[{0}]:  {1}".format(key, obj.attrs[key]))
	except:
		pass


# In[3]:


# Restore the weight
for file in os.listdir('/home/leo830227/result_deeper/saved_model/final_train_20W/'):
	if file.endswith('.meta'):
		save_file_name = file[0:-5]
		print(save_file_name)

reader = tf.train.NewCheckpointReader('/home/leo830227/result_deeper/saved_model/final_train_20W/{0}'.format(save_file_name))
all_variables = reader.get_variable_to_shape_map()

conv1_weight_r = reader.get_tensor("conv1_weight")
conv2_weight_r = reader.get_tensor("conv2_weight")
conv3_weight_r = reader.get_tensor("conv3_weight")
conv4_weight_r = reader.get_tensor("conv4_weight")

full1_weight_r = reader.get_tensor("full1_weight")
full2_weight_r = reader.get_tensor("full2_weight")
full3_weight_r = reader.get_tensor("full3_weight")

conv1_bias_r = reader.get_tensor("conv1_bias")
conv2_bias_r = reader.get_tensor("conv2_bias")
conv3_bias_r = reader.get_tensor("conv3_bias")
conv4_bias_r = reader.get_tensor("conv4_bias")

full1_bias_r = reader.get_tensor("full1_bias")
full2_bias_r = reader.get_tensor("full2_bias")
full3_bias_r = reader.get_tensor("full3_bias")


FLAGS = tf.app.flags.FLAGS
"""
tf.app.flags.DEFINE_float('nsr', None, 'NSR = 1 / SNR of training data') # NSR of training data
tf.app.flags.DEFINE_float('learning_rate_num', None, 'Learning rate of training') # Learning rate
tf.app.flags.DEFINE_integer('trainning_times', None, 'Which time of trainning')
tf.app.flags.DEFINE_integer('generation', None, '# of Training generation (one batch size)')
tf.app.flags.DEFINE_float('keep_prob_num', None, 'keep prob of dropout layer')
"""

#for fintal training
tf.app.flags.DEFINE_string('nsr', 'final', None)
tf.app.flags.DEFINE_float('learning_rate_num', 0.001, 'Learning rate of training') # Learning rate
#tf.app.flags.DEFINE_integer('trainning_times', , 'Which time of trainning')
tf.app.flags.DEFINE_integer('generation', 200, '# of Training generation (one batch size)')
tf.app.flags.DEFINE_float('keep_prob_num', 1.0, 'keep prob of dropout layer')

# new training data
#nsr = 1.0 # change

nsr = FLAGS.nsr
#snr = round(1 / nsr, 3)
#nsr = round(nsr, 1)


data_folder = '/work1/leo830227/GW_training_data_EOBNRv2_whitened_template_HDF5_mass_labeled/data_{0}_label_mass_20W_template'.format(nsr)
h5_file_name = 'data_{0}_label_mass_20W_template.h5'.format(nsr)


# In[345]:

# commen out for final training
"""
trainning_times = FLAGS.trainning_times
#trainning_times = 19
next_trainning_times = trainning_times + 1
print(next_trainning_times)
"""
next_trainning_times = 'Trainsfer'



# ####################################################################################################

# In[346]:


paths = '{0}/{1}'.format(data_folder, h5_file_name) #data path
#print(paths)


# In[5]:


# Show the information of training data.
s = timeit.default_timer()

fh = h5py.File(paths, 'r')
"""
for key in fh.keys():
	print(key) 

fh.visititems(dump_info)
"""
e = timeit.default_timer()

print(e - s,'sec')


# In[6]:


# Load the training data: training set, validation set and testing set. X for strain data, Y for labeled mass.
trainX = fh['train']['trainX'].value
trainY = fh['train']['trainY'].value
trainZ = fh['train']['train_SNR'].value
trainValX = fh['train']['trainValX'].value
trainValY = fh['train']['trainValY'].value
trainValZ = fh['train']['trainVal_SNR'].value
testX = fh['train']['testX'].value
testY = fh['train']['testY'].value
testZ = fh['train']['test_SNR'].value
fh.close()


# In[7]:


# Check the shape of training, validation and testing set
print('new_train:\t', np.shape(trainX), np.shape(trainY))#, np.shape(trainZ))
print('new_trainVal:', np.shape(trainValX), np.shape(trainValY))#, np.shape(trainValZ))
print('new_test:\t', np.shape(testX), np.shape(testY))#, np.shape(testZ))


# # Part II: 深度學習模型建製

# Parameters

# In[8]:


# Parameters of the model
batch_size = 128
evaluation_size = 100
time_seires_length = trainX[0].shape[0]

# for dropout
keep_prob_num = FLAGS.keep_prob_num
#############################################
# Learning rate and batch size有許多不同的取法，不完全都是固定的，有很多取法是可以隨著training變動的，ex. minibatch 
# approximation
#############################################

#print(time_seires_length)
target_size = 1 + 1

print(target_size)
num_channels = 1 # only one channel: strain
generation = FLAGS.generation
eval_every = 20 #幾個generation回報一次


# In[9]:


# define placeholder
x_input_shape = tf.TensorShape([None, None, num_channels]) # the shpae of input data, we don't need to specify first dimension, if do, we cannot feed x_input with test data.
x_input = tf.placeholder(tf.float32, shape = x_input_shape, name='x_input')
y_target = tf.placeholder(tf.int32, shape = (None), name='Y_target')
learning_rate = tf.placeholder(tf.float32, shape = (None), name='learning_rate')

# for validation
"""
eval_input_shape = (None, time_seires_length, num_channels)
eval_input = tf.placeholder(tf.float32, shape = eval_input_shape, name='eval_input')
eval_target = tf.placeholder(tf.int32, shape = (None), name='eval_target')
"""
#for dropot
keep_prob = tf.placeholder(tf.float32, shape = (None), name='keep_prob')

#for tensorboard
train_accuracy_tb = tf.placeholder(tf.float32, shape = (None), name='train_accuracy_tb')
#train_accuracy_tb_scalar = tf.summary.scalar('train_accuracy', train_accuracy_tb)
val_accuracy_tb = tf.placeholder(tf.float32, shape = (None), name='val_accuracy_tb')
#val_accuracy_tb_scalar = tf.summary.scalar('val_accuracy', val_accuracy_tb)
global_steps = tf.Variable(0, name='global_steps', trainable=False)


# In[10]:

# for final training
new_save_folder_name = 'Transfer_train'
new_save_file_name = 'final_train_{0}'.format(generation)

# In[11]:


# 定義variables for convolution layers
conv1_features = 64 #out_channels of conv1
conv1_kernal_size = 16
conv1_weight = tf.Variable(conv1_weight_r, name='conv1_weight') # [filter_width, in_channels, out_channels]
conv1_dilation_rate = [1]
conv1_stride = [1]

#print(np.shape(conv1_weight))

conv1_bias = tf.Variable(conv1_bias_r, name='conv1_bias')
conv1_padding = 'SAME'

conv2_features = 128 #out_channels of conv2
conv2_kernal_size = 16
conv2_weight = tf.Variable(conv2_weight_r, name='conv2_weight')
conv2_stride = [1]
conv2_dilation_rate = [2]
conv2_bias = tf.Variable(conv2_bias_r, name='conv2_bias')
conv2_padding = 'SAME'

conv3_features = 256 #out_channels of conv3
conv3_kernal_size = 16
conv3_weight = tf.Variable(conv3_weight_r, name='conv3_weight')
conv3_stride = [1]
conv3_dilation_rate = [2]
conv3_bias = tf.Variable(conv3_bias_r, name='conv3_bias')
#print(conv3_bias)
conv3_padding = 'SAME'

conv4_features = 512 #out_channels of conv3
conv4_kernal_size = 32
conv4_weight = tf.Variable(conv4_weight_r, name='conv4_weight')
conv4_stride = [1]
conv4_dilation_rate = [2]
conv4_bias = tf.Variable(conv4_bias_r, name='conv4_bias')
conv4_padding = 'SAME'


#定義variables for pooling layers
pool1_type = 'MAX'
pool1_stride = [4]
pool_size1 = 4 # N window for 1st max pool layer
pool1_padding = 'SAME'


pool2_type = 'MAX'
pool2_stride = [4]
pool_size2 = 4 # N window for 2nd max pool layer
pool2_padding = 'SAME'

pool3_type = 'MAX'
pool3_stride = [4]
pool_size3 = 4 # N window for 3nd max pool layer
pool3_padding = 'SAME'

pool4_type = 'MAX'
pool4_stride = [4]
pool_size4 = 4 # N window for 4nd max pool layer
pool4_padding = 'SAME'

# 定義variables for fully connected 

resulting_length = time_seires_length // (pool_size1 * pool_size2 * pool_size3 *  pool_size4) 
full1_input_size = resulting_length * conv4_features

#print(full1_input_size)

fully_connected_size1 = 128
full1_weight = tf.Variable(full1_weight_r, name='full1_weight')
full1_bias = tf.Variable(full1_bias_r, name='full1_bias')


fully_connected_size2 = 128
fully_connected_size3 = 64
full2_weight = tf.Variable(full2_weight_r, name='full2_weight')
full2_bias = tf.Variable(full2_bias_r, name='full2_bias')

full3_weight = tf.Variable(full3_weight_r, name='full3_weight')
full3_bias = tf.Variable(full3_bias_r, name='full3_bias')


# learning rate
learning_rate_num = FLAGS.learning_rate_num


save_folder_name = 'Trainsfer_train'

#restorer_var = tf.train.Saver(var_list={'conv1_weight:0':conv1_weight, 'conv1_bias:0':conv1_bias, 'conv2_weight:0':conv2_weight, 'conv2_bias:0':conv2_bias, 'conv3_weight:0':conv3_weight, 'conv3_bias:0':conv3_bias, 'conv4_weight:0':conv4_weight, 'conv4_bias:0':conv4_bias, 'full1_weight:0':full1_weight, 'full1_bias:0':full1_bias, 'full2_weight:0':full2_weight, 'full2_bias:0':full2_bias, 'full3_weight:0':full3_weight, 'full3_bias:0':full3_bias}

"""
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
		restorer = tf.train.import_meta_graph('/home/leo830227/result_deeper/saved_model/{0}/{1}'.format(save_folder_name, save_file_name))
		# Restore the variables
		restorer.restore(sess,tf.train.latest_checkpoint('/home/leo830227/result_deeper/saved_model/{0}'.format(save_folder_name)))
"""		


# ### Construct Deep Neuron Network
# In[12]:


def detect_GW(input_data):
	# Frist Conv-MaxPool-ReLU Layer
	conv1 = tf.nn.convolution(input_data, 
							  conv1_weight, 
							  conv1_padding,
							  strides = conv1_stride, 
							  dilation_rate = conv1_dilation_rate,
							  name = 'conv1')
	
	tf.summary.histogram('layer1/conv_weight_histogram', conv1_weight)
	
	
	max_pool1 = tf.nn.pool(input = conv1, 
						   window_shape= [pool_size1], 
						   strides = pool1_stride,
						   pooling_type= pool1_type, 
						   padding= pool1_padding,
						   name = 'max_pool1')
	
	#relu1 = tf.nn.relu(tf.nn.bias_add(max_pool1, conv1_bias))
	relu1 = tf.nn.relu(max_pool1,
					   name = 'relu1')
	
	"""
	relu1_drop = tf.nn.dropout(relu1,
							   keep_prob,
							   name = 'layer1_drop') 
	"""
	########################################################################################################
	#max_pool1 = tf.layers.MaxPooling1D(inputs=relu1, pool_size=[max_pool_size1], strides=1, padding='same',
										#data_format='channels_last')
	#####################################################################################################2##
	
	# Second Conv-MaxPool-ReLU Layer
	conv2 = tf.nn.convolution(relu1, 
							  conv2_weight, 
							  conv2_padding,
							  strides = conv2_stride, 
							  dilation_rate = conv2_dilation_rate,
							  name = 'conv2')
	tf.summary.histogram('layer2/conv_weight', conv2_weight)
	
	max_pool2 = tf.nn.pool(input = conv2, 
						   window_shape= [pool_size2], 
						   strides = pool2_stride, 
						   pooling_type= pool2_type, 
						   padding= pool2_padding,
						   name = 'max_pool2')
	
	#relu2 = tf.nn.relu(tf.nn.bias_add(max_pool2, conv2_bias))
	relu2 = tf.nn.relu(max_pool2,
					   name = 'relu2')
	"""
	relu2_drop = tf.nn.dropout(relu2,
							   keep_prob,
							   name = 'layer2_drop') 
	"""
	
	# Thrid Conv-MaxPool-ReLU Layer
	conv3 = tf.nn.convolution(relu2, 
							  conv3_weight, 
							  conv3_padding,
							  strides = conv3_stride,
							  dilation_rate = conv3_dilation_rate,
							  name = 'conv3')
	tf.summary.histogram('layer3/conv_weight', conv3_weight)
	
	max_pool3 = tf.nn.pool(input = conv3, 
						   window_shape= [pool_size3], 
						   strides = pool3_stride, 
						   pooling_type= pool3_type, 
						   padding= pool3_padding,
						   name = 'max_pool3')
	
	#relu3 = tf.nn.relu(tf.nn.bias_add(max_pool3, conv3_bias))
	relu3 = tf.nn.relu(max_pool3, name = 'relu3')
	
	
	# Fourth Conv-MaxPool-ReLU Layer
	conv4 = tf.nn.convolution(relu3, 
							  conv4_weight, 
							  conv4_padding,
							  strides = conv4_stride,
							  dilation_rate = conv4_dilation_rate,
							  name = 'conv4')
	tf.summary.histogram('layer4/conv_weight', conv4_weight)
	
	max_pool4 = tf.nn.pool(input = conv4, 
						   window_shape= [pool_size4], 
						   strides = pool4_stride, 
						   pooling_type= pool4_type, 
						   padding= pool4_padding,
						   name = 'max_pool4')
	
	#relu3 = tf.nn.relu(tf.nn.bias_add(max_pool3, conv3_bias))
	relu4 = tf.nn.relu(max_pool4, name = 'relu4')
	
	# Flaten the out put for next fully connected layer
	final_conv_shape = tf.shape(relu4)
	final_shape = final_conv_shape[1] * final_conv_shape[2]
	flat_output = tf.reshape(max_pool4, [final_conv_shape[0], final_shape])
	#print(flat_output.shape)
	# First Fully Connected Layer
	fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias), name = 'fully_connected1') # tf.matmul:矩陣相乘
	#print(full1_weight)
	#dropout
	dropout = tf.nn.dropout(fully_connected1,
							keep_prob,
							name = 'dropout')
	
	relu5 = tf.nn.relu(dropout, name = 'relu5')
	
	# Second Fully Connected Layer
	fully_connected2 = tf.add(tf.matmul(relu5, full2_weight), full2_bias, name = 'fully_connected2')
	
	relu6 = tf.nn.relu(fully_connected2, name = 'relu6')
	
	# Third Fully Connected Layer
	fully_connected3 = tf.add(tf.matmul(relu6, full3_weight), full3_bias, name = 'fully_connected3')
	
	final_model_output = fully_connected3
	
	return(final_model_output, conv1, max_pool1, relu1, conv2, max_pool2, relu2, conv3, max_pool3, relu3, conv1_weight, conv2_weight, conv3_weight)


model_output = detect_GW(x_input)
model_output_0 = model_output[0]
#print(model_output_0)
#print(y_target)
#val_model_output = detect_GW(eval_input)
#val_model_output_0 = val_model_output[0]															 


# Define loss function (which used to show how good the training result is, and also the variable NN try to minimize.)

# In[13]:


loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output_0, labels=y_target), name = 'loss')
#loss = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(model_output_0, y_target)), y_target), name = 'loss')
#val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=val_model_output_0, labels=eval_target), name = 'val_loss')
print(loss)
loss_scalar = tf.summary.scalar('loss', loss)
val_loss_scalar = tf.summary.scalar('val_loss', loss)


# In[14]:


# Create a prediction function
#prediction = model_output_0
prediction = tf.nn.softmax(model_output_0, name = 'prediction')
#print(prediction.name)

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


# In[15]:


# define the configuration of tensorflow
config = tf.ConfigProto()

# allow gpu memory grow, which will use GPU memory according to the requirement. This configuration let tensorflow
# won't occupied all of the GPU memory
config.gpu_options.allow_growth = True

# Start a graph session with the configuration
sess = tf.Session(config = config)


# In[16]:


# Create an optimizer
my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
										beta1=0.93,
										beta2=0.999,
										epsilon=1e-08,
										use_locking=False,
										name='Adam') #, name='my_optimizer_{0}'.format(term))
############################################################################################
#(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False )
############################################################################################
train_step = my_optimizer.minimize(loss)#, global_step=global_steps)#, name='train_step')
#slot = my_optimizer.get_slot_names()
#print(slot)
"""
for v in tf.trainable_variables():
	print(my_optimizer.get_slot(v, 'momentum'))
"""
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(
	'/home/leo830227/result_deeper/tensorboard/Trainsfer_train', sess.graph)
train_accuracy_tb_scalar = tf.summary.scalar('train_accuracy', train_accuracy_tb)
val_accuracy_tb_scalar = tf.summary.scalar('val_accuracy', val_accuracy_tb)
		
# Initialize Variables (Variables需要初始化才能用)
init = tf.global_variables_initializer()
sess.run(init)
# Creat a saver
saver=tf.train.Saver(max_to_keep=1)

# use tf.add_to_collection to save train_step to a collection, because train_step operation cannot saved by name directly
# it may because the function of train_step, tf.train.MomentumOptimizer, have a default name.
tf.add_to_collection('train_step', train_step)

tf.add_to_collection('loss_scalar', loss_scalar)
tf.add_to_collection('val_loss_scalar', val_loss_scalar)
tf.add_to_collection('merged', merged)
tf.add_to_collection('train_accuracy_tb_scalar', train_accuracy_tb_scalar)
tf.add_to_collection('val_accuracy_tb_scalar', val_accuracy_tb_scalar)

#tf.add_to_collection()
print(train_step.name)


# # Part III: 訓練開始

# In[17]:


start_time2 = timeit.default_timer()

# Used for record the best model
min_loss = 100 
min_val_loss = 100
report_times = 0
#f = open('/home/leo/Leo/Deep/{0}/accuracy/{1}/{2}_learning_rate_{3}_PE.txt'.format(app, save_folder_name, save_file_name, learning_rate_num),'w')
#accuracy_saver = open('/home/leo/Leo/Deep/{0}/accuracy/{1}/{2}_learning_rate_{3}_PE_accuracy_saver.txt'.format(app, save_folder_name, save_file_name, learning_rate_num),'w')

#for save the result
train_loss = []
val_loss = []
train_acc = []
val_acc = []
mass_label_save_train = []
mass_label_save_val = []
train_pred_save = []
val_pred_save = []


for i in range(generation):
	rand_index = np.random.choice(len(trainX), size=batch_size, replace=None)
	rand_x = trainX[rand_index]
	rand_x = np.expand_dims(rand_x, 2)
	rand_y = trainY[rand_index]
	rand_z = trainZ[rand_index]
	train_dict = {x_input: rand_x, y_target: rand_y, keep_prob: keep_prob_num, learning_rate: learning_rate_num}
	
	#print(rand_index)
	#print(rand_y)
	
	
	sess.run(train_step, feed_dict=train_dict)
	temp_train_loss, temp_train_preds, temp_train_loss_tb = sess.run([loss, prediction, loss_scalar], feed_dict=train_dict)
	temp_train_acc = get_accuracy(temp_train_preds, rand_y)
	
	#print(temp_train_preds)
	#print(temp_train_preds[2,:])
	#print('###############################################################')
	#print(rand_y[2,:])
	#print('###############################################################')
	#print(np.asarray(temp_train_acc)[2,:])
	#ggdaininsadsa
	
	# 每 eval_every 步 即利用trainVal的資料來看一下模型準確率
	
	eval_index = np.random.choice(len(trainValX), size=evaluation_size)
	eval_x = trainValX[eval_index]
	eval_x = np.expand_dims(eval_x, 2)
	eval_y = trainValY[eval_index]
	eval_z = trainValZ[eval_index]
	test_dict = {x_input: eval_x, y_target: eval_y, keep_prob: 1.0}
	test_loss, test_preds, test_loss_tb = sess.run([loss, prediction, val_loss_scalar], feed_dict=test_dict)

	temp_test_acc = get_accuracy(test_preds, eval_y)
	
	 # save the best model
	if temp_train_loss < min_loss:
		min_loss = temp_train_loss
		if test_loss < min_val_loss:
			min_val_loss = test_loss
			saver.save(sess,'/home/leo830227/result_deeper_detection/saved_model/{0}/{1}'.format(save_folder_name, save_file_name), global_step=i + 1)
	   ###################
	
		# Record and print results
	if (i+1) % eval_every == 0:
		#train_loss.append(temp_train_loss)
		#train_acc.append(temp_train_acc)
		#test_acc.append(temp_test_acc)
		
		
		
		acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
		acc_and_loss = [np.round(x,2) for x in acc_and_loss]
		print('Generation # {}. Train Loss: {:.2f}. Train Acc (Val Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
		
		"""
		############################################################################################
		print('Generation: {0}.'.format(i+1))
		print('Train:')
		for k in range(np.shape(rand_x)[0]):
			print('Mass Label: {0}. Prediction: {1}. acc: {2}.'.format(rand_y[k,:], temp_train_preds[k,:], temp_train_acc[k]))
		print('Loss: {0}'.format(temp_train_loss))
		
		print('############################################################################################')
		
		print('Validation:')
		for m in range(np.shape(eval_x)[0]):
			print('Mass Label: {0}. Prediction: {1}. Delta Mass: {2}.'.format(eval_y[m,:], test_preds[m,:], temp_test_acc[m]))
		print('Loss: {0}'.format(test_loss))
		############################################################################################
		"""
		# save the training result
		train_loss.append(temp_train_loss)
		val_loss.append(test_loss)
		train_pred_save.append(temp_train_preds)
		val_pred_save.append(test_preds)
		mass_label_save_train.append(rand_z)
		mass_label_save_val.append(eval_z)
		train_acc.append(temp_train_acc)
		val_acc.append(temp_test_acc)
		#f.write('Generation # {}. Train Loss: {:.2f}. Train Acc (Val Acc): {:.2f} ({:.2f})\n'.format(*acc_and_loss))
		
		#tensorboard
		summary = sess.run(merged, feed_dict = train_dict)
		train_acc_tb = sess.run(train_accuracy_tb_scalar, feed_dict= {train_accuracy_tb: temp_train_acc})
		val_acc_tb = sess.run(val_accuracy_tb_scalar, feed_dict = {val_accuracy_tb: temp_test_acc})
		writer.add_summary(summary, i)
		writer.add_summary(temp_train_loss_tb, i)
		writer.add_summary(test_loss_tb, i)
		writer.add_summary(train_acc_tb, i)
		writer.add_summary(val_acc_tb, i)
		
		report_times = report_times + 1
		
		"""
		if np.abs(temp_test_acc - temp_train_acc) > 30:
			print('Overfitting!')
			break
		else:
			continue
		"""

# reshape the result to save.
train_reshape_shape = (report_times * batch_size, -1)
val_reshape_shape = (report_times * evaluation_size, -1)
train_pred_save = list(np.reshape(np.asarray(train_pred_save), train_reshape_shape))
val_pred_save = list(np.reshape(np.asarray(val_pred_save), val_reshape_shape))
#mass_label_save_train = list(np.reshape(np.asarray(mass_label_save_train), train_reshape_shape))
#mass_label_save_val = list(np.reshape(np.asarray(mass_label_save_val), val_reshape_shape))
#train_acc = list(np.reshape(np.asarray(train_acc), train_reshape_shape))
#val_acc = list(np.reshape(np.asarray(val_acc), val_reshape_shape))
"""
print(train_pred_save)
print('############################################################################################')
print(val_acc)
"""
#f.close()
writer.close()

stop_time2 = timeit.default_timer()
print(stop_time2 - start_time2)
cost_time = stop_time2 - start_time2


# In[25]:


gh = h5py.File('/home/leo830227/result_deeper_detection/accuracy/{0}/result.h5'.format(save_folder_name),'w')
gh1 = gh.create_group('Meta')
gh2 = gh.create_group('Loss')
gh3 = gh.create_group('Prediction')
gh4 = gh.create_group('Mass_label')
#gh5 = gh.create_group('acc')
gh5 = gh.create_group('Result')

gh1['Infromation'] = 'm1 and m2 parameter estimation result.'
#gh1['SNR'] = snr
gh1['Generation'] = generation
gh1['Learning_rate'] = learning_rate_num
gh1['Keep_prob'] = keep_prob_num
#mass_label_tmp = np.vstack((trainZ, trainValZ))
#mass_label = list(np.vstack((mass_label_tmp, testY)))
#gh1['Mass_label'] = mass_label
gh1['train, validation, test #'] = [np.shape(trainX)[0], np.shape(trainValX)[0], np.shape(testX)[0]]
gh1['Training_times'] = cost_time
gh1['Handsome'] = 'Leo'

gh2['Loss_function'] = 'Mean absolute relative error function, mean(abs(true answer - prediction) / true answer)'
gh2['Train_loss'] = train_loss
gh2['Validation_loss'] = val_loss

gh3['Train_prediction'] = train_pred_save
gh3['Validation_prediction'] = val_pred_save


gh4['Train_mass_label'] = mass_label_save_train
gh4['Validation_mass_label'] = mass_label_save_val


#gh5['Function'] = 'abs(true answer - prediction) / true answer * 100, means percent.'
#gh5['Information'] = '[m1 relative loss, m2 relative loss].'
gh5['Train_acc'] = train_acc
gh5['Validation_acc'] = val_acc


# ######################################################################################

# ######################################################################################

# # Part IV: Test the Model

tf.reset_default_graph()

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
restorer = tf.train.import_meta_graph('/home/leo830227/result_deeper/saved_model/{0}/{1}'.format(save_folder_name, save_file_name))
# Restore the variables
restorer.restore(sess,tf.train.latest_checkpoint('/home/leo830227/result_deeper/saved_model/{0}'.format(save_folder_name)))


# In[27]:


# Beccause I can't load whole test et in once (it will out of memeory), so I have to redefine the 
# accuracy function
def get_num_correct_test_set(logits, targets):
	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return(num_correct)

#def get_sensitivity_test_set(logits, targets):
	


# In[28]:


# In[28]:



test_index = np.arange(0, test_size, 1)

print(max(test_index))

# Cut the test set into peices, avoiding out of memeory problem.
test_size_input = int(test_size / cut_number)
for i in range (0, cut_number):
	locals()['test_index_%s'%(i+1)] = test_index[test_size_input * i: test_size_input * (i + 1)]
#print(np.shape(test_index_2))
for i in range(1, cut_number + 1):
	locals()['test_x_%s'%i]= testX[locals()['test_index_%s'%i]]
	locals()['test_x_%s'%i] = np.expand_dims(locals()['test_x_%s'%i], 2)
	locals()['test_y_%s'%i]= testY[locals()['test_index_%s'%i]]
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

"""
print('Test:')
for k in range(np.shape(rand_x)[0]):
	print('Mass Label: {0}. Prediction: {1}. acc: {2}.'.format(test_mass_label_np[k,:], result_preds_np[k,:], test_acc_np[k,:]))
"""
# save result into HDF5 file.
cost_time2 = stop_time3 - start_time3

gh1['Test_times'] = cost_time2
gh3['Test_prediction'] = result_preds_np_list
gh4['Test_mass_label'] = test_mass_label_np_list
gh5['Test_acc'] = test_acc

gh.close()

print(stop_time3 - start_time3)




# ## 2018/3/2 
# The input data need to be normalized to -1~1 (at least not differ a lot), it can be checked by using two different inputs: '/home/leo/Leo/PyCBC/data4_sin_not_norm/' and '/home/leo/Leo/PyCBC/data3_sin_norm/'
# restore the weifgt from previous training, which was about detection.
