#!/usr/local/bin/python

#########################################################################
# 																		#
# Author: Deep Chakraborty												#
# Date Created: 18/05/2016												#
# Purpose: 1) To pickle the artificial data for use with tensorflow		#
# 		   2) To train a neural network with the loaded data 			#
# 		    		   													#
#########################################################################

import sys
import numpy as np
import tensorflow as tf
import os
import pickle
from scipy.stats import mode

################    All Constants and paths used    #####################
path = "./data/"
n_classes = 10 # Number of classes in bird data
parametersFileDir = "./data/parameters_mfcc_2.pkl"
relativePathForTest = "./data/melfilter48/test_standalone/"
testFilesExtension = '.mfcc'
confMatFileDirectory = './data/confusion_multi.txt'
classifiedTestDirectory = './testOutput/'

################    Data Loading and Plotting    ########################
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def indices(a, func):
	"""Finds elements of the matrix that correspond to a function"""
	return [i for (i, val) in enumerate(a) if func(val)]


#test_labels_dense = np.loadtxt('./data/ground_truth.txt');
#test_labels_dense = test_labels_dense.astype(int)
# test_y = dense_to_one_hot(test_labels_dense, num_classes = n_classes)
# print train_labels_dense
# plot_data(train_X, train_labels_dense)
# time.sleep(10)
# plt.close('all')
print("Data Loaded and processed ...")
################## Neural Networks Training #################################

print("Verifying Neural Network Parameters ...")

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
# n_hidden_3 = 256 # 3rd layer num features
# n_hidden_4 = 256
# n_hidden_5 = 128
n_input = 585 # input dimensionality

x = tf.placeholder("float32", [None, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    #Hidden layer with RELU activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    #Hidden layer with sigmoid activation
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
	#layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
	#layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
	#layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))
	#return tf.nn.softmax(tf.matmul(layer_2, _weights['out']) + _biases['out'])
	return tf.nn.sigmoid(tf.matmul(layer_2, _weights['out']) + _biases['out'])

print("Loading saved Weights ...")
file_ID = parametersFileDir
f = open(file_ID, "rb")
W = pickle.load(f)
b = pickle.load(f)

# print "b1 = ", b['b1']
# print "b2 = ", b['b2']
# print "b3 = ", b['out']

weights = {
	'h1': tf.Variable(W['h1']),
	'h2': tf.Variable(W['h2']),
	#'h3': tf.Variable(W['h3']),
	#'h4': tf.Variable(W['h4']),
	#'h5': tf.Variable(W['h5']),
	'out': tf.Variable(W['out'])
	}

biases = {
	'b1': tf.Variable(b['b1']),
	'b2': tf.Variable(b['b2']),
	#'b3': tf.Variable(b['b3']),
	#'b4': tf.Variable(b['b4']),
	#'b5': tf.Variable(b['b5']),
	'out': tf.Variable(b['out'])
}
# print type(b['b1'])
# print type(biases['b1'])

f.close()

# layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
# layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))


pred = multilayer_perceptron(x, weights, biases)

print("Testing the Neural Network")
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	num_examples = 0
	for root, dirs, files in os.walk(relativePathForTest, topdown=False):
		for name in dirs:
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]

			for part in parts:
				num_examples += 1

	# Test model

	# likelihood = tf.argmax(tf.reduce_mean(pred, 0),1)
	test_labels_dense = np.zeros(num_examples)
	test_labels_dense = test_labels_dense.astype(int)
	label = np.zeros(test_labels_dense.shape[0])
	ind = 0
	gt = 0
	print("Computing...")
	if len(sys.argv) == 1:
		for root, dirs, files in os.walk(relativePathForTest, topdown=False):
			list_actual_class = []
			list_predicted_class = []
			for name in dirs:
				#print(name)
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]
				directory = os.path.dirname(classifiedTestDirectory)
				if not os.path.exists(directory):
					os.makedirs(directory)

				for part in parts:
					#print("Part : ",part)

					example = np.loadtxt(os.path.join(root,name,part))
					i = 0
					rows, cols = example.shape
					context = np.zeros((rows-14,15*cols)) # 15 contextual frames
					part_mat = np.zeros((rows-14,10))
					#print(pred)
					see = tf.reduce_sum(pred,0)
					#print(see)
					while i <= (rows - 15):
						ex = example[i:i+15,:].ravel()
						ex = np.reshape(ex,(1,ex.shape[0]))
						part_mat[i:i+1,:] = sess.run(pred, feed_dict={x:ex})
						context[i:i+1,:] = ex
						i += 1
					ptemp = str(part)
					p2temp = ptemp.split('.')
					#print(p2temp)
					#np.savetxt(classifiedTestDirectory+name+p2temp[0]+'.txt', part_mat)
					# see = tf.argmax(pred, 1)
					product = np.argmax(np.asarray(see.eval({x: context})))
					#print(product)
					product_count = (np.asarray(see.eval({x: context})))		#Contains a count of windows per class
					max_product_count = np.amax(np.asarray(see.eval({x: context})))
					threshold_factor = 0.4
					avg_classes = 2
					#threshold = ((rows-14)/avg_classes)*threshold_factor
					threshold = max_product_count*threshold_factor;
					#print(threshold)
					predicted_class = []
					for j in range(n_classes):
						if(product_count[j] >= threshold):
							predicted_class.append(j+1)
					print("predicted class", predicted_class)
					actual_class_temp = name.split('_')
					#actual_class_temp.pop(0)
					actual_class = []
					for j in actual_class_temp:
						actual_class.append(int(j[1:]))
					print("actual class", actual_class)
					list_actual_class.append(actual_class)
					list_predicted_class.append(predicted_class)
					#print("product_count", product_count)
					#print("max_product_count", max_product_count)
					# product = pred.eval({x: example})
    				# sums = np.sum(product,1, keepdims = True)
    				# sumtodiv = np.tile(sums,26)
    				# prob = np.divide(product,sumtodiv)
    				# prodofprob = np.cumsum(np.log(prob),0)

    				# product = product.reshape((product.shape[0],1))
    				# print product.shape
    				# np.savetxt('./data/product.txt', product, delimiter = ' ')
    				# if i == 0:
    				# 	 np.savetxt('output.txt',np.asarray(pred.eval({x: example})), delimiter = ' ');
    				# label[ind] = product
    				# print mode(product)
    				# label[ind],_ = mode(product,axis=None)
					label[ind] = product
					test_labels_dense[ind] = gt
    				# print label.shape
					ind += 1
				gt += 1
		#print("predicted class", list_predicted_class)
		#print("actual class", list_actual_class)
		accuracy_multi = 0
		recall = 0
		precision = 0
		hamming_loss = 0
		for j in range(len(list_actual_class)):
			intersection = len(set(list_actual_class[j]) & set(list_predicted_class[j]))
			union = len(set(list_actual_class[j]) | set(list_predicted_class[j]))
			accuracy_multi += (intersection/union)
			if len(list_actual_class) > 0:
				recall += intersection/len(list_actual_class[j])
			if len(list_predicted_class) > 0:
				precision += intersection/len(list_predicted_class[j])
			hamming_loss += (union - intersection)/n_classes
		accuracy_multi = (accuracy_multi/len(list_actual_class))*100
		recall = (recall/len(list_actual_class))
		precision = (precision/len(list_actual_class))
		hamming_loss = (hamming_loss/len(list_actual_class))
		print("~ Results ~")
		print("Accuracy is %.4f " % accuracy_multi)
		print("Recall is %.4f " % recall)
		print("Precision is %.4f " % precision)
		print("Hamming Loss is %.4f " % hamming_loss)

	else:
		file_specified = sys.argv[1]
		example = np.loadtxt(file_specified)
		i = 0
		rows, cols = example.shape
		context = np.zeros((rows-14,15*cols)) # 15 contextual frames
		while i <= (rows - 15):
			ex = example[i:i+15,:].ravel()
			ex = np.reshape(ex,(1,ex.shape[0]))
			context[i:i+1,:] = ex
			i += 1
        # see = tf.argmax(pred, 1)
		see = tf.reduce_sum(pred,0)
		product = np.argmax(np.asarray(see.eval({x: context})))

		def softmax(x):
			"""Compute softmax values for each sets of scores in x."""
			e_x = np.exp(x - np.max(x))
			return e_x / e_x.sum()

		confidence_matrix = softmax(see.eval({x:context}))
		confidence_matrix = confidence_matrix*100

		file_label = product

		list1 = ['spotted_nutcracker_GHNP', 'greyhooded_warbler_GHNP', 'palerumped_warbler_GHNP', 'grey_bellied_cuckoo_GHNP', 'oriental_cuckoo_GHNP', 'lesser_cuckoo-GHNP', 'russetbacked_sparrow_GHNP', 'streaked_laughingthrush_GHNP', 'western_tragopan_GHNP', 'blackcrested_tit_GHNP', 'rufous_gorgetted_flycatcher_GHNP', 'largebilled_crow_GHNP', 'chestnutcrowned_laughingthrush_GHNP','great_barbet_GHNP','blackandyellow_grosbeak_GHNP', 'yellowbellied_fantail_GHNP','himalayan_monal_GHNP','black_throated_tit_GHNP', 'rock_bunting_GHNP','whitecheeked_nuthatch_GHNP', 'grey_bushchat_GHNP', 'orangeflanked_bushrobin_GHNP','eurasian_treecreeper_GHNP','greywinged_blackbird_GHNP','rufousbellied_niltava_GHNP','golden_bushrobin_GHNP']

		print("Predicted Class Label: ",'%s' % list1[file_label]," with confidence: " '%.4f' %confidence_matrix[file_label])

if len(sys.argv) == 1:
    label = label.astype(int)
    conf = np.zeros((n_classes,n_classes), dtype = np.int32)

    for i in range(label.shape[0]):
    	conf[test_labels_dense[i],label[i]] += 1

    # print(conf)
    np.savetxt(confMatFileDirectory, conf,fmt='%i', delimiter = ' ')
    accuracy = np.sum(np.diag(conf))
    accuracy = (float(accuracy)/label.shape[0]) * 100
    #print("Accuracy is %.4f " % accuracy)
    	# plt.savefig(pp, format='pdf')
    	# pp.close()
    # plt.show()

    # plt.close('all')
