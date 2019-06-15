import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import math
import time
import func_include_deapth
from tensorflow.python.framework import ops
from cnn_utils import *

Left_Image_Folder_Path  = 'C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Depth_mapping\\Dataset\\Left_images'
Middle_Image_Folder_Path= 'C:\\Users\bhatt\\Desktop\\Jupyter Notebook\\Depth_mapping\\Dataset\\Middle_images'
Right_Image_Folder_Path = 'C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Depth_mapping\\Dataset\\Right_images'
# Loss_file_path          = '/home/saisai/stereo_correspondence/loss.txt'
# Inference_path          = '/home/saisai/stereo_correspondence/inference'
# log_directory           = '/home/saisai/stereo_correspondence/log_directory'
# Test_Folder_Path_left   = '/home/saisai/stereo_correspondence/left_images'
# Test_Folder_Path_right  = '/home/saisai/stereo_correspondence/right_images'
# checkpoint_path         = '/home/saisai/stereo_correspondence/log_directory/stereo_correspondence/model-20'
# Test_inference_path     = '/home/saisai/stereo_correspondence/test_inference'

Epochs             = 100
NUM_INPUT_CHANNELS = 3
BatchSize 	       = 3
N_ROWS		       = 256
N_COLS		       = 512

#Image_left,Image_middle,Image_right = func_include_deapth.read_images_train()


config                          = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess 		                    = tf.Session(config=config)

image_loss_param            = 0.8
Shift_smoothness_loss_param = 0.1

def generate_image(image, Xoffset, Yoffset):         #Generate the target Image from Disparity map & offset
	hf = tf.cast(N_ROWS, dtype=tf.float32)
	wf = tf.cast(N_COLS, dtype=tf.float32)
	image_flat = tf.reshape(image,   [-1, NUM_INPUT_CHANNELS])
	x_offset   = tf.reshape(Xoffset, [-1, N_ROWS, N_COLS]    )
	y_offset   = tf.reshape(Yoffset, [-1, N_ROWS, N_COLS]    )

	xf, yf = tf.meshgrid(tf.cast(tf.range(N_COLS), tf.float32), tf.cast(tf.range(N_ROWS), tf.float32))
	xf     = tf.tile(xf, tf.stack([BatchSize, 1]))
	yf     = tf.tile(yf, tf.stack([BatchSize, 1]))
	xf     = tf.reshape(xf, [-1, N_ROWS, N_COLS])
	yf     = tf.reshape(yf, [-1, N_ROWS, N_COLS])
	xf     = xf + x_offset * wf
	yf     = yf + y_offset * hf
			
	a = tf.expand_dims(xf - tf.floor(xf), axis=-1)
	b = tf.expand_dims(yf - tf.floor(yf), axis=-1)

	xl = tf.clip_by_value(tf.cast(tf.floor(xf), dtype=tf.int32), 0, N_COLS - 1)
	yt = tf.clip_by_value(tf.cast(tf.floor(yf), dtype=tf.int32), 0, N_ROWS - 1)
	xr = tf.clip_by_value(xl + 1, 0, N_COLS - 1)
	yb = tf.clip_by_value(yt + 1, 0, N_ROWS - 1)
	batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(BatchSize), axis=-1), axis=-1), [1, N_ROWS, N_COLS])

	idx_lt    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yt * N_COLS + xl, [-1]), tf.int32)
	idx_lb    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yb * N_COLS + xl, [-1]), tf.int32)
	idx_rt    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yt * N_COLS + xr, [-1]), tf.int32)
	idx_rb    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yb * N_COLS + xr, [-1]), tf.int32)
	val = tf.zeros_like(a)   
	val += (1 - a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lt), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS]) 
	val += (1 - a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lb), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS])
	val += (0 + a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rt), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS])
	val += (0 + a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rb), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS])

	return val 

def generate_shift(image, Xoffset, Yoffset):           #To create the disparity map

	hf = tf.cast(N_ROWS, dtype=tf.float32)
	wf = tf.cast(N_COLS, dtype=tf.float32)
	image_flat = tf.reshape(image,   [-1])
	x_offset   = tf.reshape(Xoffset, [-1, N_ROWS, N_COLS]    )
	y_offset   = tf.reshape(Yoffset, [-1, N_ROWS, N_COLS]    )
	xf, yf = tf.meshgrid(tf.cast(tf.range(N_COLS), tf.float32), tf.cast(tf.range(N_ROWS), tf.float32))
	xf     = tf.tile(xf, tf.stack([BatchSize, 1]))
	yf     = tf.tile(yf, tf.stack([BatchSize, 1]))
	xf     = tf.reshape(xf, [-1, N_ROWS, N_COLS])
	yf     = tf.reshape(yf, [-1, N_ROWS, N_COLS])
	xf     = xf + x_offset * wf
	yf     = yf + y_offset * hf
			
	a = xf - tf.floor(xf)
	b = yf - tf.floor(yf)
	xl = tf.clip_by_value(tf.cast(tf.floor(xf), dtype=tf.int32), 0, N_COLS - 1)
	yt = tf.clip_by_value(tf.cast(tf.floor(yf), dtype=tf.int32), 0, N_ROWS - 1)
	xr = tf.clip_by_value(xl + 1, 0, N_COLS - 1)
	yb = tf.clip_by_value(yt + 1, 0, N_ROWS - 1)

	batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(BatchSize), axis=-1), axis=-1), [1, N_ROWS, N_COLS])

	idx_lt    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yt * N_COLS + xl, [-1]), tf.int32)
	idx_lb    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yb * N_COLS + xl, [-1]), tf.int32)
	idx_rt    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yt * N_COLS + xr, [-1]), tf.int32)
	idx_rb    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yb * N_COLS + xr, [-1]), tf.int32)
	val = tf.zeros_like(a)   
	val += (1 - a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lt), tf.float32), [-1, N_ROWS, N_COLS]) 
	val += (1 - a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lb), tf.float32), [-1, N_ROWS, N_COLS])
	val += (0 + a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rt), tf.float32), [-1, N_ROWS, N_COLS])
	val += (0 + a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rb), tf.float32), [-1, N_ROWS, N_COLS])
	return val 

def SSIM(source, target):

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	mean_source = tf.layers.average_pooling2d(source, (3, 3), (1, 1), 'VALID')
	mean_target = tf.layers.average_pooling2d(target, (3, 3), (1, 1), 'VALID')

	sigma_source  = tf.layers.average_pooling2d(source ** 2,      (3, 3), (1, 1), 'VALID') - mean_source ** 2
	sigma_target  = tf.layers.average_pooling2d(target ** 2,      (3, 3), (1, 1), 'VALID') - mean_target ** 2
	sigma_st      = tf.layers.average_pooling2d(source * target , (3, 3), (1, 1), 'VALID') - mean_source * mean_target

	SSIM_n = (2 * mean_source * mean_target + C1) * (2 * sigma_st + C2)
	SSIM_d = (mean_source ** 2 + mean_target ** 2 + C1) * (sigma_source + sigma_target + C2)

	SSIM = SSIM_n / SSIM_d

	return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def gradient_x(img):

	gx = img[:,:,:-1,:] - img[:,:,1:,:]
	return gx

def gradient_y(img):

	gy = img[:,:-1,:,:] - img[:,1:,:,:]
	return gy

def get_disparity_smoothness(disp, img):

	disp_gradients_x = gradient_x(disp)
	disp_gradients_y = gradient_y(disp)

	image_gradients_x = gradient_x(img)
	image_gradients_y = gradient_y(img)

	weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
	weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

	smoothness_x = disp_gradients_x * weights_x
	smoothness_y = disp_gradients_y * weights_y

	return tf.reduce_sum(tf.square(smoothness_x)) + tf.reduce_mean(tf.square(smoothness_y))

def create_placeholders(n_H0, n_W0, n_C0, n_H1, n_W1, n_C1):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_H1, n_W1, n_C1])    
    return X, Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    W1 = tf.get_variable("W1", [2, 2, 9, 54], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [6, 6, 54, 18], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W1", [6, 6, 18, 6], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W2", [3, 3, 16, 3], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}
    return parameters

def compute_cost(Z3, Y):
    """
    Computes the cost
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))    
    return cost

def reset_c_h(cell):
	state_c, state_h=cell.zero_state(batch_size, tf.float32)
	state_c=tf.zeros_like(state_c)
	state_h=tf.zeros_like(state_h)
	initialstates=tf.nn.rnn_cell.LSTMStateTuple(state_c,state_h)
	cell=tf.nn.rnn_cell.LSTMCell(512,initialstates)
	return cell

def create_batches(start,batchSize):
    return trainData[start:start+batchSize],trainLabel[start:start+batchSize]

def initializeLayer(inputs,outputs):
    return tf.Variable(tf.truncated_normal([inputs,outputs],stddev=0.1)), tf.Variable(tf.zeros([outputs]))

def lossCompute():
	print(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)))
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

def trainstep(alpha, crossentropy):
    return tf.train.GradientDescentOptimizer(alpha).minimize(crossentropy)

def findaccuracy(y,y_):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_prediction,accuracy

def model(iS,iM,iT,parameters=initialize_parameters(), learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):

	left_images  = os.listdir(Left_Image_Folder_Path)
	middle_images= os.listdir(Middle_Image_Folder_Path)
	right_images = os.listdir(Right_Image_Folder_Path)

	num_batches = int(len(left_images) / BatchSize)

	input_S = iS 			# Getting the source image as input
	input_M = iM
	input_T = iT 			# Getting the target image as input
	image = tf.concat([iS,(tf.concat[iM,iT])],3)     

	W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
	conv1=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(image, W1,   [1,2,2,1], padding = 'VALID', name = 'conv_layer_1' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
	conv2=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(conv1, W2,   [1,2,2,1], padding = 'VALID', name = 'conv_layer_2' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
	conv3=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(conv2, W3,   [1,3,3,1], padding = 'VALID', name = 'conv_layer_3' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
	conv4=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(conv3, W4,   [1,2,2,1], padding = 'VALID', name = 'conv_layer_3' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

	map1 = conv4[:,:,0]
	map2 = conv4[:,:,1]
	map3 = conv4[:,:,2]
	neurons=512
	cell1=tf.nn.rnn_cell.LSTMCell(neurons)
	initial_state1 = cell1.zero_state(batch_size, tf.float32)
	outputs1, state1 = tf.nn.dynamic_rnn(cell1, map1 ,dtype=tf.float32)	
	output1=tf.reshape(outputs1,[-1,512])
	cell2=tf.nn.rnn_cell.LSTMCell(neurons)
	initial_state2 = cell2.zero_state(batch_size, tf.float32)
	outputs2, state2 = tf.nn.dynamic_rnn(cell2, map2 ,dtype=tf.float32)
	output2=tf.reshape(outputs2,[-1,512])
	cell3=tf.nn.rnn_cell.LSTMCell(neurons)
	initial_state3 = cell3.zero_state(batch_size, tf.float32)
	outputs3, state3 = tf.nn.dynamic_rnn(cell3, map3 ,dtype=tf.float32)
	output3=tf.reshape(outputs3,[-1,512])
	W1,B1=initializeLayer(512,neurons)
	W2,B2=initializeLayer(neurons,1)
	W3,B3=initializeLayer(512,neurons)
	W4,B4=initializeLayer(neurons,1)
	W5,B5=initializeLayer(512,neurons)
	W6,B6=initializeLayer(neurons,1)
	y1 = tf.matmul(output1,W1) + B1
    #fy1 = tf.nn.leaky_relu(y1)
	y_pred1 = tf.matmul(W2,y1) + B2
	y2 = tf.matmul(output2,W3) + B3
    #fy1 = tf.nn.leaky_relu(y1)
	y_pred2 = tf.matmul(W4,y1) + B4
	y2 = tf.matmul(output3,W5) + B5
    #fy1 = tf.nn.leaky_relu(y1)
	y_pred3 = tf.matmul(W6,y1) + B6

	# return y_pred1,y_pred2,y_pred3
	XShift_source = y_pred1
	XShift_middle = y_pred2
	XShift_target = y_pred3
	YShift        = tf.zeros_like(XShift_source)

# def loss():
	T_gen                             = generate_image(input_S,  XShift_target, YShift)
	M_gen							  =	generate_image(input_M, -XShift_middle, YShift) 		
	S_gen                             = generate_image(input_T, -XShift_source, YShift)

	target_reconstruction_loss        = tf.reduce_mean(tf.abs(T_gen - input_T)) 
	middle_reconstruction_loss        = tf.reduce_mean(tf.abs(M_gen - input_M)) 
	source_reconstruction_loss        = tf.reduce_mean(tf.abs(S_gen - input_S))

	image_reconstruction_loss         = target_reconstruction_loss + source_reconstruction_loss	+middle_reconstruction_loss		 # L1 reconstruction loss for both source and target

	SSIM_loss_target                  = tf.reduce_mean(SSIM(T_gen, input_T))
	SSIM_loss_middle                  = tf.reduce_mean(SSIM(M_gen, input_M))
	SSIM_loss_source                  = tf.reduce_mean(SSIM(S_gen, input_S))

	SSIM_loss                         = SSIM_loss_target + SSIM_loss_source + SSIM_loss_middle								 # SSIM loss for both source and target
	image_loss                        = (1 - image_loss_param) * image_reconstruction_loss + (image_loss_param) * SSIM_loss

	XShift_source_gen                 = generate_shift(XShift_target, -XShift_source, YShift)
	XShift_middle_gen                 = generate_shift(XShift_middle, -XShift_middle, YShift)
	XShift_target_gen                 = generate_shift(XShift_source, XShift_target,YShift)

	XShift_source_reconstruction_loss = tf.reduce_mean(tf.abs(XShift_source - XShift_source_gen))
	XShift_middle_reconstruction_loss = tf.reduce_mean(tf.abs(XShift_middle - XShift_middle_gen))
	XShift_target_reconstruction_loss = tf.reduce_mean(tf.abs(XShift_target -XShift_target_gen))

	XShift_consistency_loss           = XShift_source_reconstruction_loss +XShift_middle_reconstruction_loss+ XShift_target_reconstruction_loss 	 # XShift consistency loss

	XShift_source_smoothness          = get_disparity_smoothness(tf.expand_dims(XShift_source, axis = -1), sinput_S)
	XShift_middle_smoothness          = get_disparity_smoothness(tf.expand_dims(XShift_middle, axis = -1), sinput_M)
	XShift_target_smoothness          = get_disparity_smoothness(tf.expand_dims(XShift_target, axis = -1), input_T)

	XShift_smoothness_loss            = XShift_source_smoothness + XShift_target_smoothness + XShift_middle_smoothness

	XShift_smoothness_loss            = Shift_smoothness_loss_param * XShift_smoothness_loss

	total_loss = image_loss + XShift_consistency_loss + 0.1 * XShift_smoothness_loss
	

# 	return total_loss,T_gen,M_gen,S_gen,image_reconstruction_loss,XShift_consistency_loss,XShift_smoothness_loss

# def model(image1,image2,image3, Y_train1,Y_train2,Y_train3, learning_rate = 0.009,
#           num_epochs = 100, minibatch_size = 64, print_cost = True):
    """    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3            
    # m=5400
    # X_train = tf.concat([image1,(tf.concat[image2,image3])],0)     

    # n_H0=512
    # n_W0=512
    # n_C0 = 9             # to keep results consistent (numpy seed)
    # n_H1=512
    # n_W1=512
    # n_C1 = 3                         
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    # X, Y = create_placeholders(512, 512, 9, 512, 512, 3)
    
    #image=X
    # Initialize parameters
    parameters = initialize_parameters()    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    # Z3 = forward_propagation(X_train, parameters)  
    # y_pred1,y_pred2,y_pred3 = model_lstm(Z3)

    # Cost function: Add cost function to tensorflow graph
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:        
        # Run the initialization
        sess.run(init)       
        # Do the training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = 200 # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            Epoch_Loss         = 0 # Initialize the total epoch loss ss
			Average_Batch_Loss = 0 # Initialising average batch loss
			Total_Batch_Loss   = 0 # Initialising total batch loss

			Total_per_pixel_l2_loss = 0
			Total_consistency_loss  = 0
			Total_smoothness_loss   = 0
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            imageid = 0

            for minibatch in range(num_batches):
                # Select a minibatch
				Train_Source =  Image_left[imageid : imageid + BatchSize] 	#Getting the current source training images
				Train_Middle =  Image_middle[imageid : imageid + BatchSize] 	#Getting the current source training images
				Train_Target = Image_right[imageid : imageid + BatchSize]	#Getting the current target training images                # IMPORTANT: The line that runs the graph on a minibatch.
                
				imageid = imageid + BatchSize

				Image_Source = func_include_deapth.read_images_train(Train_Source, Left_Image_Folder_Path, N_ROWS, N_COLS)    # Reading the current source train Images(Normalized) and resize
				Image_Middle = func_include_deapth.read_images_train(Train_Source, Left_Image_Folder_Path, N_ROWS, N_COLS)    # Reading the current source train Images(Normalized) and resize
				Image_Target = func_include_deapth.read_images_train(Train_Target, Right_Image_Folder_Path, N_ROWS, N_COLS)    # Reading the current target train Images(Normalized) and resize


                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
			_, Batch_Loss, generated_source_image, generated_target_image, per_pixel_l2_loss, consistency_loss, smoothness_loss = sess.run((train_, loss, image_s, image_t, img_loss, cons_loss, ss_loss), feed_dict = {source : Image_Source,middle : Image_Middle, target : Image_Target, learning_rate : lr}) #Training the network with current input
                
                minibatch_cost += Batch_Loss / num_minibatches
                
                Epoch_Loss              = Epoch_Loss + Batch_Loss 
				Total_Batch_Loss        = Total_Batch_Loss + Batch_Loss
				Total_per_pixel_l2_loss = Total_per_pixel_l2_loss + per_pixel_l2_loss
				Total_consistency_loss  = Total_consistency_loss + consistency_loss
				Total_smoothness_loss   = Total_smoothness_loss + smoothness_loss


				if(minibatch % 10 == 0 and minibatch > 0):

					Average_Batch_Loss        = Total_Batch_Loss / (minibatch + 1)
					average_per_pixel_l2_loss = Total_per_pixel_l2_loss / (minibatch + 1)
					average_consistency_loss  = Total_consistency_loss / (minibatch + 1)
					average_smoothness_loss   = Total_smoothness_loss / (minibatch + 1)
					
					content = str('current epoch : ' + str(e) + ', after ' + str(minibatch) + ' batches, the average batch loss is : ' + str.format('{0:.4f}', Average_Batch_Loss) + ' and per pixel reconstruction loss is ' + str.format('{0:.4f}', average_per_pixel_l2_loss) + ', consistency loss is ' + str.format('{0:.4f}', average_consistency_loss) + ', smoothness loss ' + str.format('{0:.4f}', average_smoothness_loss))

					loss_file.write(content)
					loss_file.write(os.linesep)
					print(content)

				if(minibatch % 100 == 0 and minibatch > 0):

					image_name = Inference_path + '/' + 'epoch_' + str(e) + '_batch_' + str(minibatch) + '_' + Train_Target[0]
					out_image  = np.vstack((np.hstack((Image_Source[0] * 255.0, Image_Target[0] * 255.0)), np.hstack((generated_source_image[0, :, :, :] * 255.0, generated_target_image[0, :, :, :] * 255.0))))
					cv2.imwrite(image_name, out_image)
            # Print the cost every epoch

            if epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            # if epoch % 1 == 0:
                costs.append(minibatch_cost)

        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        # predict_op = tf.argmax(Z3, 1)
        # correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print(accuracy)
        # train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        # test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # print("Train Accuracy:", train_accuracy)
        # print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters

source = tf.placeholder(dtype = tf.float32, shape = (BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS), name = 'source_input')
middle = tf.placeholder(dtype = tf.float32, shape = (BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS), name = 'middle_input')
target = tf.placeholder(dtype = tf.float32, shape = (BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS), name = 'target_input')

model  = model(source, middle, target)






