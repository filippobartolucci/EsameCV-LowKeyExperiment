import os 
from align.align_trans import get_reference_facial_points, warp_and_crop_face
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from align.detector import detect_faces
from PIL import Image
import numpy as np
import torch 

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def get_model(name, path, input_size=(112,112)):
	"""
	Loads a model from the specified path.

	:param name: A string representing the name of the model.
		Valid values are 'ir50', 'ir101', 'ir152', 'irse50', 'irse101', and 'irse152'.
	:param path: A string representing the path to the model.
	:param input_size: A tuple representing the size of the input image.
		The default value is (112,112).
	:return: A PyTorch model.
	"""

	models = {
		'ir50': IR_50,
		'ir101': IR_101,
		'ir152': IR_152,
		'irse50': IR_SE_50,
		'irse101': IR_SE_101,
		'irse152': IR_SE_152
	}

	if name not in models:
		raise Exception('Model name not found')
	
	model = models[name](input_size)
	return model.load_state_dict(torch.load(path, map_location=device)).to(device)
	

def clear_dir(dir_root='./imgs'):
	"""
	Clears the directory of all images that end with '_attacked.png' or '_small.png'

	:param dir_root: the directory to clear
	:return: None
	"""
	
	if not os.path.exists(dir_root):
		raise Exception('Directory does not exist')

	for img_name in os.listdir(dir_root):
		if img_name.endswith('_attacked.png') or img_name.endswith('_small.png') or img_name.endswith('_face.png'):
			os.remove(os.path.join(dir_root, img_name))

def get_cropped_face(img, crop_size = 112, scale = 1.0):
	"""
	Crops the face from an input image.

	:param img: A NumPy array representing the input image.
		The image should be in RGB format with dimensions (height, width, channels)
		or (channels, height, width).
	:param crop_size: An integer representing the size of the cropped face in pixels.
		The default value is 112.
	:param scale: A float representing the scale factor to apply to the input image.
		The default value is 1.0.
	:return: A NumPy array representing the cropped face.
		The output image will be in RGB format with dimensions (height, width, channels)
		or (channels, height, width), depending on the format of the input image.
	"""
	reference = get_reference_facial_points(default_square = True) * scale
	h,w,c = np.array(img).shape

	_, landmarks = detect_faces(img)
	facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
	face, _ = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
	
	return Image.fromarray(face)
