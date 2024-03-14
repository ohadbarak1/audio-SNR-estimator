from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
#from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys
import os
import json




'''
def crossval_CNN2D (pfile, train_data, train_labels, nsplits, sep_str, test_data=None, test_labels=None):

	# read hyperparameters from par file
	nclasses		= int(get_sep_his_par (pfile, 'nclasses'))
	epochs			= int(get_sep_his_par (pfile, 'epochs'))
	batch_size		= int(get_sep_his_par (pfile, 'batch_size'))
	verbose			= int(get_sep_his_par (pfile, 'verbose'))
	NNmodel_function	= get_sep_his_par (pfile, 'NNmodel_function')

	# reshape training data for single channel 2D CNN.
	# by default, the channels are on the fast axis, and currently I'm only using one channel.
	train_data = train_data.reshape (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)

	# convert labels to an 'nclasses' long array of output neurons.
	one_hot_labels  = to_categorical (train_labels, num_classes=nclasses)

	# Array for saving metrics of training categorical accuracy over splits.
	cat_acc		= np.zeros([nsplits], dtype=np.float32)	

	# initialize KFold object for cross-validation
	seed = 13
	np.random.seed(seed)
	kfold = StratifiedKFold (n_splits=nsplits, shuffle=True, random_state=seed)

	# if test data is provided
	do_test =  False
	if (test_data is not None and test_labels is not None):
		test_data = test_data.reshape (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
		test_labels = test_labels.flatten(order='K')
		test_prec = np.zeros([nsplits], dtype=np.float32)	
		test_recall = np.zeros([nsplits], dtype=np.float32)	
		test_specificity = np.zeros([nsplits], dtype=np.float32)	
		do_test = True

	# apply K-fold cross validation
	isplit=0
	for train_split, valid_split in kfold.split(train_data, train_labels):
		# build network
		model = globals()[NNmodel_function] (pfile, train_data)

		# Train the model
		history_callback = model.fit(train_data[train_split], one_hot_labels[train_split], epochs=epochs, batch_size=int(batch_size), verbose=verbose)

		# evaluate the loss function and accuracy on the validation data
		score = model.evaluate (train_data[valid_split], one_hot_labels[valid_split], batch_size=int(batch_size), verbose=verbose)

		# predict on the test data
		if (do_test):
			pred = model.predict(test_data, batch_size=int(batch_size), verbose=verbose)
			# The output of the classifier with the 'softmax' activation is an array with nclasses elements.
			# The class label is set to be the index of each array element.
			# The value in each array element is the probability that the input data belong to the class represented by the element's index.
			# All probabilites sum to 1.
			# We take the index of the element with maximum probability as the predicted class of each observation.
			pred = np.argmax (pred, axis=1)

			pred = pred.flatten(order='K')
			pred_metrics = precision_recall_avg (test_labels, pred)[0]

		for i, metric_name in enumerate(model.metrics_names):
			print ("%s=%f"%(metric_name, score[i]))
		print ("\n")

		cat_acc[isplit]	= score[1]
		if (do_test):
			test_prec[isplit] = pred_metrics['Average'][0]
			test_recall[isplit] = pred_metrics['Average'][1]
			test_specificity[isplit] = pred_metrics['Average'][2]

		isplit = isplit+1
		del model

	avg_cat_acc = np.mean(cat_acc)
	if (do_test):
		avg_test_prec = np.mean(test_prec)
		avg_test_recall = np.mean(test_recall)
		avg_test_specificity = np.mean(test_specificity)
		res_str = sep_str.join([str(avg_cat_acc), str(avg_test_prec), str(avg_test_recall), str(avg_test_specificity)])
	else:
		res_str = str(avg_cat_acc)

	print (res_str)

	return res_str

def train_eval_CNN2D (pfile, train_data, train_labels, nsplits, sep_str, test_data=None, test_labels=None):

	# read hyperparameters from par file
	activation		= get_sep_his_par (pfile, 'activation')
	loss			= get_sep_his_par (pfile, 'loss')
	nfullyconnected		= int(get_sep_his_par (pfile, 'nfullyconnected'))
	nclasses		= int(get_sep_his_par (pfile, 'nclasses'))
	momentum		= float(get_sep_his_par (pfile, 'momentum'))
	decay			= float(get_sep_his_par (pfile, 'decay'))
	nesterov		= int(get_sep_his_par (pfile, 'nesterov'))
	epochs			= int(get_sep_his_par (pfile, 'epochs'))
	lmbda			= float(get_sep_his_par (pfile, 'lambda'))
	verbose			= int(get_sep_his_par (pfile, 'verbose'))
	metrics			= get_sep_his_par (pfile, 'metrics')
	pooling_size	= int(get_sep_his_par (pfile, 'pooling_size'))
	dropout			= float(get_sep_his_par (pfile, 'dropout'))
	stride1			= int(get_sep_his_par (pfile, 'stride1'))
	dilation1		= int(get_sep_his_par (pfile, 'dilation1'))
	stride2			= int(get_sep_his_par (pfile, 'stride2'))
	dilation2		= int(get_sep_his_par (pfile, 'dilation2'))

	# read hyperparameter lists from par file
	lr_list				= get_sep_his_par (pfile, 'lr').split(',')
	batch_size_list		= get_sep_his_par (pfile, 'batch_size').split(',')
	nconv_list			= get_sep_his_par (pfile, 'nconv').split(',')
	n1_conv_list		= get_sep_his_par (pfile, 'n1_conv').split(',')
	n2_conv_list		= get_sep_his_par (pfile, 'n2_conv').split(',')

	# reshape training data for single channel 2D CNN.
	# by default, the channels are on the fast axis, and currently I'm only using one channel.
	train_data = train_data.reshape (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)

	# convert labels to an 'nclasses' long array of output neurons.
	one_hot_labels  = to_categorical (train_labels, num_classes=nclasses)

	# Array for saving metrics of training categorical accuracy over splits.
	cat_acc		= np.zeros([nsplits], dtype=np.float32)	

	# initialize KFold object for cross-validation
	seed = 13
	np.random.seed(seed)
	kfold = StratifiedKFold (n_splits=nsplits, shuffle=True, random_state=seed)

	# if test data is provided
	do_test =  False
	if (test_data is not None and test_labels is not None):
		test_data = test_data.reshape (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
		test_prec = np.zeros([nsplits], dtype=np.float32)	
		test_recall = np.zeros([nsplits], dtype=np.float32)	
		test_specificity = np.zeros([nsplits], dtype=np.float32)	
		do_test = True

	res = {}
	itest = 0

	# iterate over parameter sets, apply K-fold cross validation for each parameter set.
	for lr in lr_list:
		for batch_size in batch_size_list:
			for nconv in nconv_list:
				for n2_conv in n2_conv_list:
					for n1_conv in n1_conv_list:
						isplit=0
						for train_split, valid_split in kfold.split(train_data, train_labels):
							# build network
							model = Sequential()
							# input layer
							model.add(Conv2D(int(nconv), (int(n2_conv), int(n1_conv)),
									  activation=activation,
									  padding='valid',
									  strides=(stride2, stride1),
									  dilation_rate=(dilation2, dilation1),
									  input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3])))
							#model.add(BatchNormalization(axis=-1))
							model.add(AveragePooling2D(pool_size=(pooling_size, pooling_size)))
							model.add(Flatten())
							model.add(Dropout(dropout))
							model.add(Dense(nfullyconnected,
									  activation=activation,
									  kernel_regularizer=regularizers.l2(lmbda)))
							# final "one-hot" layer
							model.add(Dense(nclasses,
											activation='softmax',
											kernel_regularizer=regularizers.l2(lmbda)))
						
							#print model.summary()
						
							#sgd = SGD (lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
							#model.compile(optimizer=sgd,
							adam = Adam (lr=float(lr), beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
							model.compile(optimizer=adam,
										  loss=loss,
										  metrics=[metrics])
					
							# Train the model
							history_callback = model.fit(train_data[train_split], one_hot_labels[train_split], epochs=epochs, batch_size=int(batch_size), verbose=verbose)
					
							# evaluate the loss function and accuracy on the validation data
							score = model.evaluate (train_data[valid_split], one_hot_labels[valid_split], batch_size=int(batch_size), verbose=verbose)
					
							# predict on the test data
							if (do_test):
								pred = model.predict(test_data, batch_size=int(batch_size), verbose=verbose)
								# The output of the classifier with the 'softmax' activation is an array with nclasses elements.
								# The class label is set to be the index of each array element.
								# The value in each array element is the probability that the input data belong to the class represented by the element's index.
								# All probabilites sum to 1.
								# We take the index of the element with maximum probability as the predicted class of each observation.
								pred = np.argmax (pred, axis=1)

								pred = pred.flatten(order='K')
								test_labels = test_labels.flatten(order='K')
								pred_metrics = precision_recall_avg (test_labels, pred)[0]

							#print "\nsplit=%d, lr=%s, batch_size=%s, nconv=%s, n1_conv=%s, n2_conv=%s"%(isplit, lr, batch_size, nconv, n1_conv, n2_conv)
							#for i, metric_name in enumerate(model.metrics_names):
							#	print "%s=%f"%(metric_name, score[i])
							#print "\n"

							cat_acc[isplit]	= score[1]
							if (do_test):
								test_prec[isplit] = pred_metrics['Average'][0]
								test_recall[isplit] = pred_metrics['Average'][1]
								test_specificity[isplit] = pred_metrics['Average'][2]

							isplit = isplit+1
							del model

						avg_cat_acc = np.mean(cat_acc)
						if (do_test):
							avg_test_prec = np.mean(test_prec)
							avg_test_recall = np.mean(test_recall)
							avg_test_specificity = np.mean(test_specificity)
							res_str = sep_str.join([lr, batch_size, nconv, n1_conv, n2_conv, str(avg_cat_acc), str(avg_test_prec), str(avg_test_recall), str(avg_test_specificity)])
						else:
							res_str = sep_str.join([lr, batch_size, nconv, n1_conv, n2_conv, str(avg_cat_acc)])

						print res_str
						res[str(itest)] = res_str
						itest += 1

	return res
'''

class SNREstimator ():
	def __init__(self, json_path):
		with open(json_path, "r") as f:
			self.network_params = json.load(f)
			self.input_shape = None

	def train_model (self, checkpoint_path, train_data, train_labels, valid_data=None, valid_labels=None):
		train_data = train_data.reshape (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
		self.input_shape=[train_data.shape[1], train_data.shape[2], 1]

		# load model if the file already exists.
		# otherwise, build a new model based on the provided function name
		if (os.path.exists(checkpoint_path)):
			print (f"loading existing model from path: {checkpoint_path}")
			model = load_model(checkpoint_path)
		else:
			print (f"building new model")
			model = self.build_model ()

		valid_tuple = None
		#provide validation data and labels as a tuple
		if (valid_data is not None and valid_labels is not None):
			if valid_data.shape[1] != train_data.shape[1] or valid_data.shape[2] != train_data.shape[2]:
				raise Exception('dimension mismatch between training and validation data')
			
			valid_data = valid_data.reshape (valid_data.shape[0], valid_data.shape[1], valid_data.shape[2], 1),
			valid_tuple = (valid_data, valid_labels)

			checkpoint_monitor='val_loss'
		else:
			checkpoint_monitor='loss'

		# add checkpoints for trained model.
		# Save only the "best" model according to the value of validation data loss, 
		# or traiing data loss if validation data are not supplied
		my_callbacks = [
			#ModelCheckpoint(filepath='%s.epoch={epoch:03d}.val_loss={val_loss:2.4f}.h5'%(checkpoint_path.split('.')[0]),
			ModelCheckpoint(filepath=checkpoint_path,
							monitor=checkpoint_monitor,
							save_best_only=True,
							verbose=self.network_params["input"]["verbose"])
		]
		# Train the model
		history_callback = model.fit(train_data, train_labels,
							   epochs=self.network_params["input"]["epochs"],
							   batch_size=self.network_params["input"]["batch_size"],
							   validation_data=valid_tuple,
							   shuffle=True,
							   callbacks=my_callbacks,
							   verbose=self.network_params["input"]["verbose"])

		# save model to file
		model.save (checkpoint_path)
		return history_callback

	def infer(self, checkpoint_path, test_data):
		# read hyperparameters from par file
		batch_size	= self.network_params["input"]["batch_size"]
		verbose		= self.network_params["input"]["verbose"]
		
		if test_data.shape[1] != self.input_shape[0] or test_data.shape[2] != self.input_shape[1]:
			raise Exception('dimension mismatch between test and training data')
		test_data = test_data.reshape (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
		
		# load model from file
		model = load_model (checkpoint_path)
		# run inference on test data using the loaded model
		pred = model.predict (test_data, batch_size=batch_size, verbose=verbose)
		return pred
	
	def build_model (self):
		model = Sequential()

		# Input Shape: If data_format="channels_first": A 4D tensor with shape: (batch_size, channels, height, width)

		for i, layer in enumerate (self.network_params["input"]["nn_hyperparams"]["layers"]):
			if i == 0:
				if layer["layer_type"] != "Conv":
					raise ValueError ('Expecting the first layer in the network to be convolutional')
			
			if "name" in layer.keys():
				layer_name = layer["name"]

			if layer["layer_type"] == "Conv":
				filters = layer["filters"]
				kernel_size = layer["kernel_size"]
				stride = layer["strides"]
				padding = layer["padding"]
				dilation_rate = layer["dilation_rate"]
				activation = layer["activation"]

				if i == 0:
					model.add(Conv2D (filters, (kernel_size[0], kernel_size[1]),
						activation=activation, padding=padding,
						strides=(stride[0], stride[1]), dilation_rate=(dilation_rate[0], dilation_rate[1]),
						data_format="channels_last",
						input_shape=self.input_shape))
				else:
					model.add(Conv2D (filters, (kernel_size[0], kernel_size[1]),
						activation=activation, padding=padding,
						strides=(stride[0], stride[1]), dilation_rate=(dilation_rate[0], dilation_rate[1])))
		
			elif layer["layer_type"] == "MaxPool":
				stride = layer["strides"]
				padding = layer["padding"]
				pool_size = layer["pool_size"]
				model.add(MaxPooling2D (pool_size=(pool_size[0], pool_size[1]),
					strides=(stride[0],stride[1]), padding=padding) )
			
			elif layer["layer_type"] == "AvgPool":
				stride = layer["strides"]
				padding = layer["padding"]
				pool_size = layer["pool_size"]
				model.add(AveragePooling2D (pool_size=(pool_size[0], pool_size[1]),
					strides=(stride[0],stride[1]), padding=padding) )
			
			elif layer["layer_type"] == "Dropout":
				drop = layer["drop"]
				model.add(Dropout(drop))

			elif layer["layer_type"] == "Flatten":
				model.add(Flatten())
			
			elif layer["layer_type"] == "Dense":
				units = layer["units"]
				activation = layer["activation"]
				if "regularizer" in layer.keys() and "lambda" in layer.keys():
					
					lmbda = layer["lambda"]
					if layer["regularizer"] == "l2":
						model.add(Dense(units, activation=activation,
					      kernel_regularizer=regularizers.l2(lmbda)))
					elif layer["regularizer"] == "l1":
						model.add(Dense(units, activation=activation,
					      kernel_regularizer=regularizers.l1(lmbda)))
					else:
						raise ValueError ("unsupported regularizer type '{}'".format(layer["regularizer"]))
				else:
					model.add(Dense(units, activation=activation))
			else:
				raise ValueError ("unsupported layer type '{}'".format(layer["layer_type"]))
			
		print (model.summary())

		adam = Adam (learning_rate=self.network_params["input"]["learning_rate"],
			beta_1=self.network_params["input"]["adam_beta1"],
			beta_2=self.network_params["input"]["adam_beta2"],
			amsgrad=False)
		
		model.compile(optimizer=adam,
				loss=self.network_params["input"]["loss"],
				metrics=[self.network_params["input"]["metrics"]])
		
		return model


# return classification precision and recall per class
def precision_recall (y_true, y_pred):

	res = []
	y_unique = np.unique(y_true)

	for label in y_unique:
		precision=0.; recall=0.; F1=0.; jaccard=0.; specificity=0.
		TP=0; TN=0; FN=0; FP=0; 
		zip_y = zip (y_true, y_pred)

		TP = sum (int (yt==label and yp==label) for yt, yp in zip_y)
		TN = sum (int (yt!=label and yp!=label) for yt, yp in zip_y)
		FP = sum (int (yt!=label and yp==label) for yt, yp in zip_y)
		FN = sum (int (yt==label and yp!=label) for yt, yp in zip_y)

		if (TP+FP > 0):
			precision	= float(TP) / (TP+FP)
		if (TP+FN > 0):
			recall		= float(TP) / (TP+FN)
		if (TP+FP+FN > 0):
			jaccard		= float(TP) / (TP+FP+FN)
		if (TN+FP > 0):
			specificity	= 1. - float(FP) / (TN+FP)
		if (precision+recall > 0):
			F1 = 2.*precision*recall / (precision+recall)


		metrics = {str(label): [precision, recall, F1, jaccard, specificity]}
		res.append (metrics)

	return res

# return weighted average of classification precision and recall
def precision_recall_avg (y_true, y_pred):

	res = []
	y_unique, y_counts = np.unique(y_true, return_counts=True)
	y_dict=dict(zip(y_unique, y_counts))
	ny = y_true.size

	precision=0.; recall=0.; F1=0.; jaccard=0.; specificity=0.

	for label in y_dict.keys():
		TP=0; TN=0; FN=0; FP=0; 
		zip_y = zip (y_true, y_pred)

		TP = sum (int (yt==label and yp==label) for yt, yp in zip_y)
		TN = sum (int (yt!=label and yp!=label) for yt, yp in zip_y)
		FP = sum (int (yt!=label and yp==label) for yt, yp in zip_y)
		FN = sum (int (yt==label and yp!=label) for yt, yp in zip_y)

		if (TP+FP > 0):
			precision	+= float(TP) / (TP+FP) * y_dict[label]
		if (TP+FN > 0):
			recall		+= float(TP) / (TP+FN) * y_dict[label]
		if (TP+FP+FN > 0):
			jaccard		+= float(TP) / (TP+FP+FN) * y_dict[label]
		if (TN+FP > 0):
			specificity	+= (1. - float(FP) / (TN+FP)) * y_dict[label]
		if (precision+recall > 0):
			F1 += 2.*precision*recall / (precision+recall)

	F1 = 2.*(precision/ny * recall/ny) / (precision/ny + recall/ny)

	metrics = {'Average': [precision/ny, recall/ny, F1, jaccard/ny, specificity/ny]}
	res.append (metrics)
	return res

# calculate confusion matrix
def confusion_matrix (y_true, y_pred):

	y_unique = np.unique(y_true)

	CM = np.zeros ([len(y_unique), len(y_unique)], dtype=np.float32)

	# labels may not be consecutive index numbers, so use a dictionary to convert labels to indices in confusion matrix 
	class_dict={}
	for i, y in enumerate(y_unique):
		class_dict[y] = i

	for label in zip (y_true, y_pred):
		CM[class_dict[label[1]]][class_dict[label[0]]] += float (label[0]==label[1])
		CM[class_dict[label[1]]][class_dict[label[0]]] += float (label[0]!=label[1])

	return CM

# return arrays of true positive and false positives per class
def true_false_positive_perclass (y_true, y_pred):

	y_unique = np.unique(y_true)

	TP = np.empty ([len(y_unique), len(y_true)], dtype=np.float32)
	FP = np.empty ([len(y_unique), len(y_true)], dtype=np.float32)

	for iy, y in enumerate(y_unique):
		for ipred, label in enumerate(zip (y_true, y_pred)):
			TP[iy][ipred] = float (label[0]==y and label[1]==y)
			FP[iy][ipred] = float (label[0]!=y and label[1]==y)

	return TP, FP

# return arrays of total true positives and false positives
def true_false_positive_total (y_true, y_pred):

	TP = np.empty ([len(y_true)], dtype=np.float32)
	FP = np.empty ([len(y_true)], dtype=np.float32)

	for ipred, label in enumerate(zip (y_true, y_pred)):
		TP[ipred] = float (label[0]==label[1])
		FP[ipred] = float (label[0]!=label[1])

	return TP, FP

# calculate true positive rate, false positive rate, precision and F1 per class.
# Also calculate the micro-average and macro-average of each metric.
def ROC_poly (y_true_poly, y_pred_poly, activation, n_op, o_op, d_op):

	if y_true_poly.shape[0] != y_pred_poly.shape[1]:
		print ("ROC_poly: Number of observations in true and predicted labels must be equal")
		sys.exit(1)
	if y_true_poly.shape[1] != y_pred_poly.shape[2]:
		print ("ROC_poly: Number of classes in true and predicted labels must be equal")
		sys.exit(1)
	if y_true_poly.shape[0] != activation.shape[0]:
		print ("ROC_poly: Number of observations in true labels and activations must be equal")
		sys.exit(1)
	if y_true_poly.shape[1] != activation.shape[1]:
		print ("ROC_poly: Number of classes in true labels and activations must be equal")
		sys.exit(1)

	nclasses = y_true_poly.shape[1]

	TPR = np.zeros([nclasses+2,n_op], dtype=np.float)
	FPR = np.zeros([nclasses+2,n_op], dtype=np.float)
	precision = np.zeros([nclasses+2,n_op], dtype=np.float)
	F1 = np.zeros([nclasses+2,n_op], dtype=np.float)
	loss = np.zeros([nclasses+1], dtype=np.float)

	# iterate over operating thresholds and class index to calculate metrics for each individual class
	for ip,op in enumerate(np.arange(o_op, (n_op-1)*d_op, d_op)):
		for ic in xrange(nclasses):
			diff = y_pred_poly[ip,:,ic] - y_true_poly[:,ic]
			mult = y_pred_poly[ip,:,ic] * y_true_poly[:,ic]

			# example of logic of the next section:
			# case:				[TP  TN  FP  FN]
			# predicted labels: [1   0   1   0]
			# true labels:      [1   0   0   1]
			# diff(pred-true):	[0   0   1   -1]
			# mult(pred*true):	[1   0   0   0]

			FP = len(diff[diff == 1])
			FN = len(diff[diff == -1])
			TP_TN = len(diff[diff == 0])
			TP = len(mult[mult == 1])
			TN = TP_TN - TP

			if TP+FN > 0.:
				TPR[ic,ip] = float(TP) / (TP+FN) # recall = true positive rate
			if FP+TN > 0.:
				FPR[ic,ip] = float(FP) / (FP+TN) # false positive rate
			if TP+FP > 0.:
				precision[ic,ip]	= float(TP) / (TP+FP)
			if precision[ic,ip] + TPR[ic,ip] > 0.:
				F1[ic,ip] = 2.*precision[ic,ip]*TPR[ic,ip] / (precision[ic,ip]+TPR[ic,ip])

			del diff, mult

	# iterate over operating thresholds to generate micro-average of metrics for all classes combined
	for ip,op in enumerate(np.arange(o_op, (n_op-1)*d_op, d_op)):
		diff = y_pred_poly[ip,:,:] - y_true_poly[:,:]
		mult = y_pred_poly[ip,:,:] * y_true_poly[:,:]

		# example of logic of the next section:
		# case:				[TP  TN  FP  FN]
		# predicted labels: [1   0   1   0]
		# true labels:      [1   0   0   1]
		# diff(pred-true):	[0   0   1   -1]
		# mult(pred*true):	[1   0   0   0]

		FP = len(diff[diff == 1])
		FN = len(diff[diff == -1])
		TP_TN = len(diff[diff == 0])
		TP = len(mult[mult == 1])
		TN = TP_TN - TP

		if TP+FN > 0.:
			TPR[nclasses,ip] = float(TP) / (TP+FN) # recall = true positive rate
		if FP+TN > 0.:
			FPR[nclasses,ip] = float(FP) / (FP+TN) # false positive rate
		if TP+FP > 0.:
			precision[nclasses,ip]	= float(TP) / (TP+FP)
		if precision[nclasses,ip] + TPR[nclasses,ip] > 0.:
			F1[nclasses,ip] = 2.*precision[nclasses,ip]*TPR[nclasses,ip] / (precision[nclasses,ip]+TPR[nclasses,ip])

		del diff, mult

	# calculate macro-average of each metric
	for ip,op in enumerate(np.arange(o_op, (n_op-1)*d_op, d_op)):
		TPR[nclasses+1,ip] = TPR[0:nclasses,ip].mean()
		FPR[nclasses+1,ip] = FPR[0:nclasses,ip].mean()
		precision[nclasses+1,ip] = precision[0:nclasses,ip].mean()
		F1[nclasses+1,ip] = F1[0:nclasses,ip].mean()

	# calculate binary cross-entropy loss
	for ic in xrange(nclasses):
		loss[ic] = -np.mean(y_true_poly[:,ic] * np.log2(activation[:,ic]+np.finfo(np.float32).eps) + (1-y_true_poly[:,ic]) * np.log2(1-activation[:,ic]+np.finfo(np.float32).eps))

	loss[nclasses] = np.mean(loss[0:nclasses])

	return TPR, FPR, precision, F1, loss

