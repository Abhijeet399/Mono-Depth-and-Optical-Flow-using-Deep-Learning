import numpy as np 
import os
import tensorflow as tf
import cv2
import Other_Funcs

# Start the graph

Left_Image_Folder_Path  = '/home/saisai/stereo_correspondence/left_images'
Right_Image_Folder_Path = '/home/saisai/stereo_correspondence/right_images'
Loss_file_path          = '/home/saisai/stereo_correspondence/loss.txt'
Inference_path          = '/home/saisai/stereo_correspondence/inference'
log_directory           = '/home/saisai/stereo_correspondence/log_directory'
Test_Folder_Path_left   = '/home/saisai/stereo_correspondence/left_images'
Test_Folder_Path_right  = '/home/saisai/stereo_correspondence/right_images'
checkpoint_path         = '/home/saisai/stereo_correspondence/log_directory/stereo_correspondence/model-20'
Test_inference_path     = '/home/saisai/stereo_correspondence/test_inference'

Epochs             = 50
NUM_INPUT_CHANNELS = 3
BatchSize 	       = 2
N_ROWS		       = 256
N_COLS		       = 512

config                          = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess 		                    = tf.Session(config=config)

class Correspondence():
	"""
	The Correspondence class, with the network architecture, loss functions etc
	"""

	def __init__(self, iS, iT):

		# constructor that takes source and target images as inputs to generate XShift and YShift as outputs

		self.image_loss_param            = 0.8
		self.Shift_smoothness_loss_param = 0.1
		
		def generate_image(image, Xoffset, Yoffset):

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

		def generate_shift(image, Xoffset, Yoffset):

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


		################################################################################################################################################################

		self.input_S = iS 			# Getting the source image as input
		self.input_M = iM 
		self.input_T = iT 			# Getting the target image as input

		################################################################################################################################################################

		# Network construction

		self.model_input = tf.concat([self.input_S,(tf.concat([self.input_M, self.input_T], 3))],3)

		with tf.variable_scope('conv_layers_1_to_16') as scope:
			tf.set_random_seed(1)                              # so that your "random" numbers match ours
		    self.W1 = tf.get_variable("W1", [2, 2, 9, 54], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))
		    self.W2 = tf.get_variable("W2", [6, 6, 54, 18], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
		    self.W3 = tf.get_variable("W1", [6, 6, 18, 6], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))
		    self.W4 = tf.get_variable("W2", [3, 3, 16, 3], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

		    self.conv1=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(self.model_input, W1,   [1,2,2,1], padding = 'VALID', name = 'conv_layer_1' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
			self.conv2=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(self.conv1, self.W2,   [1,2,2,1], padding = 'VALID', name = 'conv_layer_2' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
			self.conv3=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(self.conv2, self.W3,   [1,3,3,1], padding = 'VALID', name = 'conv_layer_3' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
			self.conv4=tf.nn.max_pool(tf.nn.leaky_relu(tf.nn.conv2d(self.conv3, self.W4,   [1,2,2,1], padding = 'VALID', name = 'conv_layer_3' ), alpha = 0.1),ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

			self.conv_out = tf.reshape(self.conv16, (BatchSize, -1))
			
		with tf.variable_scope('fully_connected_layers') as scope:

			self.fc1	  = tf.nn.leaky_relu(tf.layers.dense(self.conv_out, units = 1024, bias_initializer = tf.initializers.ones(), name = 'dense1'), alpha = 0.5)
			self.fc2      = tf.nn.leaky_relu(tf.layers.dense(self.fc1,      units = 1024, bias_initializer = tf.initializers.ones(), name = 'dense2'), alpha = 0.5)

		with tf.variable_scope('de_conv_layers_1_to_16') as scope:

			self.deconv_in   = tf.reshape(self.fc2, (BatchSize, 2, 4, -1))

			self.deconv1     = tf.nn.leaky_relu(tf.layers.conv2d_transpose(  self.deconv_in,  512,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_1'  ), alpha = 0.1) #(N_ROWS/128, N_COLS/128)
			self.concat1     = tf.concat([self.deconv1, self.conv15],  axis = 3)
			self.deconv2_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat1,               512,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_1'    ), alpha = 0.1)   

			self.deconv2     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv2_in,  512,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_2'  ), alpha = 0.1) 
			self.concat2     = tf.concat([self.deconv2, self.conv14],  axis = 3)
			self.deconv3_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat2,               512,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_2'    ), alpha = 0.1) 

			self.deconv3     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv3_in,  256,   (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_3'  ), alpha = 0.1) #(N_ROWS/64, N_COLS/64)
			self.concat3     = tf.concat([self.deconv3, self.conv13],  axis = 3)
			self.deconv4_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat3,               256,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_3'    ), alpha = 0.1)

			self.deconv4     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv4_in,  256,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_4'  ), alpha = 0.1) 
			self.concat4     = tf.concat([self.deconv4, self.conv12],  axis = 3)
			self.deconv5_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat4,               256,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_4'    ), alpha = 0.1)

			self.deconv5     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv5_in,  128,   (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_5'  ), alpha = 0.1) #(N_ROWS/32, N_COLS/32) 
			self.concat5     = tf.concat([self.deconv5, self.conv11],  axis = 3)
			self.deconv6_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat5,               128,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_5'    ), alpha = 0.1) 
 
			self.deconv6     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv6_in,  128,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_6'  ), alpha = 0.1) 
			self.concat6     = tf.concat([self.deconv6, self.conv10],  axis = 3)
			self.deconv7_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat6,               128,   (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_6'    ), alpha = 0.1)  			

			self.deconv7     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv7_in,  64,    (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_7'  ), alpha = 0.1) #(N_ROWS/16, N_COLS/16)
			self.concat7     = tf.concat([self.deconv7, self.conv9],   axis = 3)
			self.deconv8_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat7,               64,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_7'    ), alpha = 0.1)  

			self.deconv8     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv8_in,  64,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_8'  ), alpha = 0.1) 
			self.concat8     = tf.concat([self.deconv8, self.conv8],   axis = 3)
			self.deconv9_in  = tf.nn.leaky_relu(tf.layers.conv2d( self.concat8,               64,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_8'    ), alpha = 0.1)  

			self.deconv9     = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv9_in,  32,    (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_9'  ), alpha = 0.1) #(N_ROWS/8, N_COLS/8) 
			self.concat9     = tf.concat([self.deconv9, self.conv7],   axis = 3)
			self.deconv10_in = tf.nn.leaky_relu(tf.layers.conv2d( self.concat9,               32,    (3, 3), (1, 1), padding  = 'same', name = 'deconv-out-9'    ), alpha = 0.1)  

			self.deconv10    = tf.nn.leaky_relu(tf.layers.conv2d_transpose(self.deconv10_in,  32,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_10' ), alpha = 0.1) 
			self.concat10    = tf.concat([self.deconv10, self.conv6],  axis = 3)
			self.deconv11_in = tf.nn.leaky_relu(tf.layers.conv2d(self.concat10,               32,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_10'   ), alpha = 0.1)  

			self.deconv11    = tf.nn.leaky_relu(tf.layers.conv2d_transpose(self.deconv11_in,  16,    (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_11' ), alpha = 0.1) #(N_ROWS/4, N_COLS/4) 
			self.concat11    = tf.concat([self.deconv11, self.conv5],  axis = 3)
			self.deconv12_in = tf.nn.leaky_relu(tf.layers.conv2d(self.concat11,               16,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_11'   ), alpha = 0.1)  		

			self.deconv12    = tf.nn.leaky_relu(tf.layers.conv2d_transpose(self.deconv12_in,  16,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_12' ), alpha = 0.1) 
			self.concat12    = tf.concat([self.deconv12, self.conv4],  axis = 3)
			self.deconv13_in = tf.nn.leaky_relu(tf.layers.conv2d(self.concat12,               16,    (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_12'   ), alpha = 0.1)  

			self.deconv13    = tf.nn.leaky_relu(tf.layers.conv2d_transpose(self.deconv13_in,  8,     (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_13' ), alpha = 0.1) #(N_ROWS/2, N_COLS/2) 
			self.concat13    = tf.concat([self.deconv13, self.conv3],  axis = 3)
			self.deconv14_in = tf.nn.leaky_relu(tf.layers.conv2d(self.concat13,               8,     (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_13'   ), alpha = 0.1)  

			self.deconv14    = tf.nn.leaky_relu(tf.layers.conv2d_transpose(self.deconv14_in,  8,     (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_14' ), alpha = 0.1) 
			self.concat14    = tf.concat([self.deconv14, self.conv2],  axis = 3)
			self.deconv15_in = tf.nn.leaky_relu(tf.layers.conv2d(self.concat14,               8,     (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_14'   ), alpha = 0.1)  

			self.deconv15    = tf.nn.leaky_relu(tf.layers.conv2d_transpose(self.deconv15_in,  8,     (3, 3), (2, 2), padding  = 'same', name = 'deconv_layer_15' ), alpha = 0.1) #(N_ROWS, N_COLS) 
			self.concat15    = tf.concat([self.deconv15, self.conv1],  axis = 3)
			self.deconv16_in = tf.nn.leaky_relu(tf.layers.conv2d(self.concat15,               8,     (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_15'   ), alpha = 0.1)  

			self.deconv16    = tf.nn.leaky_relu(tf.layers.conv2d_transpose( self.deconv16_in, 8,     (3, 3), (1, 1), padding  = 'same', name = 'deconv_layer_16' ), alpha = 0.1)

			self.deconv_out  = tf.nn.tanh(tf.layers.conv2d(self.deconv16,                     2,     (3, 3), (1, 1), padding  = 'same', name = 'deconv_out_16'   ), name = 'correspondence_output')  

			################################################################################################################################################################

		# Outputs

		self.XShift_source = self.deconv_out[:, :, :, 0]
		self.XShift_target = self.deconv_out[:, :, :, 1]
		self.YShift        = tf.zeros_like(self.XShift_source)

		#################################################################################################################################################################

		# Building loss functions

		self.T_gen                             = generate_image(self.input_S,  self.XShift_target, self.YShift)
		self.S_gen                             = generate_image(self.input_T, -self.XShift_source, self.YShift)

		self.target_reconstruction_loss        = tf.reduce_mean(tf.abs(self.T_gen - self.input_T)) 
		self.source_reconstruction_loss        = tf.reduce_mean(tf.abs(self.S_gen - self.input_S))

		self.image_reconstruction_loss         = self.target_reconstruction_loss + self.source_reconstruction_loss			 # L1 reconstruction loss for both source and target

		self.SSIM_loss_target                  = tf.reduce_mean(SSIM(self.T_gen, self.input_T))
		self.SSIM_loss_source                  = tf.reduce_mean(SSIM(self.S_gen, self.input_S))

		self.SSIM_loss                        = self.SSIM_loss_target + self.SSIM_loss_source 								 # SSIM loss for both source and target

		self.image_loss                        = (1 - self.image_loss_param) * self.image_reconstruction_loss + (self.image_loss_param) * self.SSIM_loss

		self.XShift_source_gen                 = generate_shift(self.XShift_target, -self.XShift_source, self.YShift)
		self.XShift_target_gen                 = generate_shift(self.XShift_source,  self.XShift_target, self.YShift)

		self.XShift_source_reconstruction_loss = tf.reduce_mean(tf.abs(self.XShift_source - self.XShift_source_gen))
		self.XShift_target_reconstruction_loss = tf.reduce_mean(tf.abs(self.XShift_target - self.XShift_target_gen))

		self.XShift_consistency_loss           = self.XShift_source_reconstruction_loss + self.XShift_target_reconstruction_loss 	 # XShift consistency loss

		self.XShift_source_smoothness          = get_disparity_smoothness(tf.expand_dims(self.XShift_source, axis = -1), self.input_S)
		self.XShift_target_smoothness          = get_disparity_smoothness(tf.expand_dims(self.XShift_target, axis = -1), self.input_T)

		self.XShift_smoothness_loss            = self.XShift_source_smoothness + self.XShift_target_smoothness

		self.XShift_smoothness_loss            = self.Shift_smoothness_loss_param * self.XShift_smoothness_loss

		#################################################################################################################################################################

		# Total loss

		self.total_loss = self.image_loss + self.XShift_consistency_loss + 0.1 * self.XShift_smoothness_loss

		#################################################################################################################################################################

#########################################################################################################################################################################

# train function

def train():

	left_images  = os.listdir(Left_Image_Folder_Path)
	right_images = os.listdir(Right_Image_Folder_Path)

	num_batches = int(len(left_images) / BatchSize)
	loss_file = open(Loss_file_path, 'a')

	train_saver = tf.train.Saver()
	train_saver.restore(sess, checkpoint_path)

	lr = 0.00005 

	for e in range(0, Epochs): #Iterating over the Epoch

		print ('In Epoch: ' + str(e))

		Epoch_Loss         = 0 # Initialize the total epoch loss ss
		Average_Batch_Loss = 0 # Initialising average batch loss
		Total_Batch_Loss   = 0 # Initialising total batch loss

		Total_per_pixel_l2_loss = 0
		Total_consistency_loss  = 0
		Total_smoothness_loss   = 0

		imageid = 0

		for Batch_ID in range(0, num_batches): #Iterating over the complete training dataset
			
			#print(Batch_ID)

			Train_Source =  left_images[imageid : imageid + BatchSize] 	#Getting the current source training images
			Train_Target = right_images[imageid : imageid + BatchSize]	#Getting the current target training images
			
			#print(Train_Source)
			#print(Train_Target)

			imageid = imageid + BatchSize

			Image_Source = Other_Funcs.read_images_train(Train_Source, Left_Image_Folder_Path, N_ROWS, N_COLS)    # Reading the current source train Images(Normalized) and resize
			Image_Target = Other_Funcs.read_images_train(Train_Target, Right_Image_Folder_Path, N_ROWS, N_COLS)    # Reading the current target train Images(Normalized) and resize

			_, Batch_Loss, generated_source_image, generated_target_image, per_pixel_l2_loss, consistency_loss, smoothness_loss = sess.run((train_, loss, image_s, image_t, img_loss, cons_loss, ss_loss), feed_dict = {source : Image_Source, target : Image_Target, learning_rate : lr}) #Training the network with current input

			Epoch_Loss              = Epoch_Loss + Batch_Loss
			Total_Batch_Loss        = Total_Batch_Loss + Batch_Loss
			Total_per_pixel_l2_loss = Total_per_pixel_l2_loss + per_pixel_l2_loss
			Total_consistency_loss  = Total_consistency_loss + consistency_loss
			Total_smoothness_loss   = Total_smoothness_loss + smoothness_loss

			if(Batch_ID % 10 == 0 and Batch_ID > 0):

				Average_Batch_Loss        = Total_Batch_Loss / (Batch_ID + 1)
				average_per_pixel_l2_loss = Total_per_pixel_l2_loss / (Batch_ID + 1)
				average_consistency_loss  = Total_consistency_loss / (Batch_ID + 1)
				average_smoothness_loss   = Total_smoothness_loss / (Batch_ID + 1)
				
				content = str('current epoch : ' + str(e) + ', after ' + str(Batch_ID) + ' batches, the average batch loss is : ' + str.format('{0:.4f}', Average_Batch_Loss) + ' and per pixel reconstruction loss is ' + str.format('{0:.4f}', average_per_pixel_l2_loss) + ', consistency loss is ' + str.format('{0:.4f}', average_consistency_loss) + ', smoothness loss ' + str.format('{0:.4f}', average_smoothness_loss))

				loss_file.write(content)
				loss_file.write(os.linesep)
				print(content)

			if(Batch_ID % 100 == 0 and Batch_ID > 0):

				image_name = Inference_path + '/' + 'epoch_' + str(e) + '_batch_' + str(Batch_ID) + '_' + Train_Target[0]
				out_image  = np.vstack((np.hstack((Image_Source[0] * 255.0, Image_Target[0] * 255.0)), np.hstack((generated_source_image[0, :, :, :] * 255.0, generated_target_image[0, :, :, :] * 255.0))))
				cv2.imwrite(image_name, out_image)


		content = str('In Epoch: ' + str(e) + ' the average loss is : ' + str(Epoch_Loss / num_batches))
		loss_file.write(content)
		loss_file.write(os.linesep)
		print (content)  # Printing the average epoch loss

		if(e % 10 == 0 and e != 0):
			train_saver.save(sess, log_directory + '/' + 'stereo_correspondence_ramanan' + '/model-after20', global_step = e) 	

		if (e == 30):
			lr = lr / 2 

		if (e == 40):
			lr = lr / 2
	
	loss_file.close() 
	
#########################################################################################################################################################################

def test():

	Train_Source = []
	Train_Target = []

	Train_Source.append('001823.png')
	Train_Target.append('001823.png') 
	Train_Source.append('001823.png')
	Train_Target.append('001823.png')

	Image_Source = Other_Funcs.read_images_train(Train_Source, Test_Folder_Path_left,  N_ROWS, N_COLS)
	Image_Target = Other_Funcs.read_images_train(Train_Target, Test_Folder_Path_right, N_ROWS, N_COLS)

	# Load model here
	train_saver = tf.train.Saver()
	train_saver.restore(sess, checkpoint_path)



	source_xshift, target_xshift, source_gen, target_gen = sess.run((out1, out2, out3, out4), feed_dict = {source : Image_Source, target : Image_Target})

	source_disp = np.zeros((BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS))
	target_disp = np.zeros((BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS))

	for i in range(NUM_INPUT_CHANNELS):

		source_disp[:, :, :, i] = source_xshift 
		target_disp[:, :, :, i] = target_xshift

	image_name = Test_inference_path + '/' + Train_Source[0]

	print(Image_Source[0].shape)
	print(source_disp.shape)
	print(source_gen.shape)

	out_image = np.vstack((np.hstack((Image_Source[0] * 255.0, Image_Target[0] * 255.0)), np.hstack((source_disp[0, :, :, :] * 512.0 * 10, target_disp[0, :, :, :] * 512.0 * 10)), np.hstack((source_gen[0, :, :, :] * 255.0, target_gen[0, :, :, :] * 255.0))))
	"""
	have to post process properly
	"""
	cv2.imwrite(image_name, out_image)


#########################################################################################################################################################################

# main function

# Defining placeholders for input source and target

source = tf.placeholder(dtype = tf.float32, shape = (BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS), name = 'source_input')
target = tf.placeholder(dtype = tf.float32, shape = (BatchSize, N_ROWS, N_COLS, NUM_INPUT_CHANNELS), name = 'target_input')

learning_rate = tf.placeholder(tf.float32, shape = [])

# feeding the input source and target to the model

model  = Correspondence(source, target)

# loss calculation

loss      = model.total_loss
image_t   = model.T_gen
image_s   = model.S_gen
img_loss  = model.image_reconstruction_loss
cons_loss = model.XShift_consistency_loss
ss_loss   = model.XShift_smoothness_loss

out1 = model.XShift_source
out2 = model.XShift_target
out3 = model.S_gen
out4 = model.T_gen 

# optimisation

#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.9, momentum = 0.95, epsilon = 1e-6)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = 0.9, beta2 = 0.999)
train_= optimizer.minimize(loss)

init_op = tf.global_variables_initializer()		#Initializing the global variables
sess.run(init_op)

#train()
test()