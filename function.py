from PIL import Image
import numpy as np
import os
import random

def init_list(size):
	all_images = os.listdir('%d/gray' % size)
	random.shuffle(all_images)
	return all_images

def load(start, number, size, img_list):
	end = start + number
	all_images = img_list[start:end]
	res_gray = np.zeros((number, size, size, 1))
	res_color = np.zeros((number, size, size, 3))
	for i, image in enumerate(all_images):
		image_gray = Image.open("%d/gray/%s" % (size, image))
		image_color = Image.open("%d/color/%s" % (size, image))
		
		array_gray = np.array(image_gray)
		array_color = np.array(image_color)

		image_gray.close()
		image_color.close()
		
		res_gray[i,:,:,0] = array_gray[:,:]
		res_color[i,:,:,:] = array_color[:,:,:]
	return res_gray, res_color
