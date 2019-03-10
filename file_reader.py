import os
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
from utils import *

rootdir = "salicon"
i = 0
imagelist = []
outputlist = []
mode = 'L'
crop_size = (128,128)

for subdir, dirs, files in os.walk(rootdir):

	i +=1
	for file in files:
		if file.endswith(".jpg") or file.endswith(".jpeg"):
			filename = os.path.basename(file)
			print(filename)
			img = imresize(imread(subdir + '/' + filename, mode=mode), crop_size)
			if "SaliencyMap" in filename:
				outputlist.append(img)
			else:
				imagelist.append(img)

		imagelist = np.array(imagelist)
		outputlist = np.array(outputlist)
		print(imagelist.shape)
		print(outputlist.shape)
		imagelist = list(imagelist)
		outputlist = list(outputlist)

save_array(imagelist, "test_images_FILEREADER2")
save_array(outputlist, "test_outputs_FILEREADER2")