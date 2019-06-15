import cv2
import numpy as np
from PIL import Image 

def read_images_train(path_images, path_folder, N_ROWS, N_COLS):
	#This function is to read the images, resize them and normalize them

	Image_List=[]  #Empty List to store Read and Normalzied Images

	for i in path_images: 

		image_=Image.open(path_folder+'/'+i)
		image_=image_.resize((N_COLS,N_ROWS))
		image_=np.asarray(image_)
		Image_List.append(image_/255.0)

	return Image_List	
