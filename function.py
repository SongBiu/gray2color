from PIL import Image
import numpy as np
import os

def load(**kwargs):
	start = kwargs['start']
	end = kwargs['start']+kwargs['number']
	size = kwargs['size']
	all_images = os.listdir("%d/gray" % size)[start:end]
	res_gray = np.zeros((kwargs['number'], size, size, 1))
	res_color = np.zeros((kwargs['number'], size, size, 3))
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
