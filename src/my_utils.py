from align.align_trans import get_reference_facial_points, warp_and_crop_face
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from util.prepare_utils import prepare_models, prepare_dir_vec, get_ensemble, prepare_data
import torchvision.transforms as transforms
from align.detector import detect_faces
from util.attack_utils import  Attack
from PIL import Image
import numpy as np
import torch 
import natsort
import os 

import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
to_tensor = transforms.ToTensor()

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
	model.load_state_dict(torch.load(path, map_location=device))
	return model.to(device)
	
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
 
def crop_dataset(source_dir = "./my_dataset/original", face_dir =  "./my_dataset/faces"):
	"""
	Crops the faces from the images in the source directory and saves them in the face directory.

	:param source_dir: A string representing the path to the source directory.
		The default value is "./my_dataset/original".
	:param face_dir: A string representing the path to the face directory.
		The default value is "./my_dataset/faces".
	:return: None
	"""
	# Check source dir exist
	if not os.path.exists(source_dir):
		raise Exception('Source directory does not exist')

	if not os.path.exists(face_dir):
		os.mkdir(face_dir)

	# for each dir in source_dir
	for dir_name in os.listdir(source_dir):
		if not os.path.isdir(os.path.join(source_dir, dir_name)):
			continue
		# create a dir in face_dir
		if not os.path.exists(os.path.join(face_dir, dir_name)):
			os.mkdir(os.path.join(face_dir, dir_name))
		# for each img in dir
		for img_name in os.listdir(os.path.join(source_dir, dir_name)):

			if img_name.endswith(".DS_Store"):
				continue

			# crop the face and save it in the new dir
			print("Processing image {}".format(dir_name + "/" + img_name), end = '\n')
			img = Image.open(os.path.join(source_dir, dir_name, img_name)).convert("RGB")
			img = get_cropped_face(img)
			img.save(os.path.join(face_dir, dir_name, img_name))

def get_dataset(source_dir):
	"""
	This function receives a source directory and returns two lists: X and Y.
	X contains the paths to all images in the source directory.
	Y contains the corresponding labels for each image in X.

	:param source_dir: A string representing the path to the source directory.
	:return: Two lists: X and Y.
	"""
	X = []
	Y = []

	for root, dirs, files in os.walk(source_dir):
		for file in natsort.natsorted(files):
			if file.endswith(".DS_Store"):
				continue
			X.append(os.path.join(root, file))
			Y.append(os.path.basename(root))

	return X, Y

def my_split(X, Y, split = "1"):
	X1 = []
	Y1 = []

	X2 = []
	Y2 = []

	for i, _ in enumerate(X):
		if X[i].endswith(split + ".jpg") or X[i].endswith(split + "_attacked.png"):
			X1.append(X[i])
			Y1.append(Y[i])
		else:
			X2.append(X[i])
			Y2.append(Y[i])

	return X1, Y1, X2, Y2


def attack_dataset(source_dir, target_dir, models=None, model_paths=None, input_size=(112, 112), batch = 4, kernel_size_gf = 7, sigma_gf = 3):
	"""
	Attacks the images in the source directory and saves them in the target directory.

	:param source_dir: A string representing the path to the source directory. It contains directories with images to attack.
	:param target_dir: Target directory path that has the same structure as the source directory.
	:param models: A list of strings representing the names of the models to use for the attack.
		Valid values are 'ir50', 'ir101', 'ir152', 'irse50', 'irse101', and 'irse152'.
	:param model_paths: A list of strings representing the paths to the models to use for the attack.
	:param input_size: A tuple representing the size of the input image.
		The default value is (112,112).
	:return: None
	"""
	# Check source dir exist
	if not os.path.exists(source_dir):
		raise Exception('Source directory does not exist')

	if not os.path.exists(target_dir):
		os.mkdir(target_dir)

	if models is None or model_paths is None:
		print("Using default models")

		models = [
			'IR_152', 
			'IR_152', 
			'ResNet_152', 
			'ResNet_152'
		]

		model_paths = [
			'models/Backbone_IR_152_Arcface_Epoch_112.pth', 
			'models/Backbone_IR_152_Cosface_Epoch_70.pth',
			'models/Backbone_ResNet_152_Arcface_Epoch_65.pth', 
			'models/Backbone_ResNet_152_Cosface_Epoch_68.pth'
		]

	# Configuration 
	eps = 0.05
	n_iters = 50
	attack_type = 'lpips'
	c_tv = None
	c_sim = 0.05
	lr = 0.0025
	net_type = 'alex'
	noise_size = 0.005
	n_starts = 1
	combination = True
	using_subspace = False
	V_reduction_root = './'
	direction = 1


	models_attack, V_reduction, dim = prepare_models(
		models,
		input_size,
		model_paths,
		kernel_size_gf,
		sigma_gf,
		combination,
		using_subspace,
		V_reduction_root
	)

	dim = 512

	imgs_paths, _ = get_dataset(source_dir)	
	idx = 0

	while idx < len(imgs_paths):
		print("Processing batch {} of {}".format(idx // batch + 1, len(imgs_paths) // batch + 1), end = '\r')

		batch_paths = imgs_paths[idx:idx+batch]
		idx += batch

		reference = get_reference_facial_points(default_square = True)
		tensor_img = torch.cat([to_tensor(Image.open(i).convert("RGB")).unsqueeze(0) for i in batch_paths], 0).to(device)
		
		# find direction vector
		dir_vec_extractor = get_ensemble(models = models_attack, sigma_gf = None, kernel_size_gf = None, combination = False, V_reduction = V_reduction, warp = False, theta_warp = None)
		dir_vec = prepare_dir_vec(dir_vec_extractor, tensor_img, dim, combination)

		img_attacked = tensor_img.clone()
		attack = Attack(models_attack, dim, attack_type, eps, c_sim, net_type, lr,
            n_iters, noise_size, n_starts, c_tv, sigma_gf, kernel_size_gf,
            combination, warp=False, theta_warp=None, V_reduction = V_reduction)

		img_attacked = attack.execute(tensor_img, dir_vec, direction).detach().cpu()

		for i, img in enumerate(img_attacked):
			img = transforms.ToPILImage()(img)
			# given souce_dir and bath_paths[i], get relative path
			relative_path = batch_paths[i].split('/')[-2] + '/' + batch_paths[i].split('/')[-1]
			label_dir = os.path.join(target_dir, batch_paths[i].split('/')[-2])
			if not os.path.exists(label_dir):
				os.mkdir(label_dir)
			save_path = label_dir + "/{}_attacked.png".format(batch_paths[i].split('/')[-1][:-4])
			img.save(save_path)


def extract_features(model, dataset, batch_size=4):
	X = []
	for i in range(0, len(dataset), batch_size):
		batch = dataset[i:i+batch_size]
		batch = torch.cat([to_tensor(Image.open(x)).unsqueeze(0) for x in batch], dim=0)
		batch = batch.to(device)
		with torch.no_grad():
			batch_features = model(batch)
			X.append(batch_features.cpu().numpy())
	X = np.concatenate(X, axis=0)
	return X


def mixset(X_set1, X_set2, Y_set1, Y_set2, n):
	mixed_x = []
	mixed_y = []
	n = 5-n

	for i in range(len(X_set1)):
		m = int(X_set1[i].split("/")[-1].split(".")[0].split("_")[0])
		if (m > n): 
			mixed_x.append(X_set1[i])
			mixed_y.append(Y_set1[i])
		else:
			mixed_x.append(X_set2[i])
			mixed_y.append(Y_set2[i])

	return mixed_x, mixed_y
