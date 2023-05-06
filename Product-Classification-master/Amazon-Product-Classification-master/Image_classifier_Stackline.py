import pandas as pd
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os
from tqdm import tqdm
import scipy
from pylab import *

import tensorflow as tf
import random

img_size = 128
MODEL_NAME = 'ImageClassifier_Stackline.model'
df = pd.read_csv("training_data.csv", encoding = 'ISO-8859-1')

def generaete_catagories(cat_path):
	catagories = []
	with open(cat_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			catagories.append(line)
	return catagories

def label_img(word_label):
	name = word_label[0]
	headers = ['Headphones\n', 'Cables\n', 'Security & Surveillance\n', 'Streaming Media\n', 'Television Accessories\n', 'Monitor Risers\n', 'Gaming Accessories\n', 'Video Games\n', 'Video Cameras\n', '3D Printers & Supplies\n', 'Drones\n', 'Mice\n', 'Computer Accessories\n', 'Keyboards\n', 'Monitor Mounts\n', 'Monitors\n', 'Office Electronics\n', 'Camera Accessories\n', 'Range Extenders\n', 'Ink & Toner\n', 'Car & Vehicle Electronics\n', 'Video Projectors\n', 'Tablet Accessories\n', 'Car Subwoofers & Amplifiers\n', 'Tablets\n', 'Laptop Accessories\n', 'Tripods & Monopods\n', 'Televisions\n', 'Batteries\n', 'Desktops\n', 'Laptops\n', 'Home Audio\n', 'GPS & Navigation\n', 'Radar Detectors\n', 'Mobile Phone Accessories\n', 'Headsets\n', 'Binoculars & Scopes\n', 'Modems\n', 'Cases & Screen Protectors\n', 'TV Mounts & Stands\n', 'eBook Readers & Accessories\n', 'Computer Data Storage\n', 'Portable Audio & Speakers\n', 'Power Management\n', 'Computer Components\n', 'Video Cards\n', 'Printers & Scanners\n', 'Memory Cards & Flash Drives\n', 'Unlocked Cell Phones\n', 'Wearable Technology\n', 'Motherboards\n', 'Telescopes\n', 'Routers & Networking\n', 'Car Dash Cams\n', 'Microphones and Accessories\n', 'Two Way Radios\n', 'Blu-ray and DVD Players\n', 'Standing Desks\n', 'Cameras\n', 'Switches\n', 'Calculators\n', 'Camera Lenses\n', 'Game Consoles\n']
	headers = list(map(lambda s: s.strip(), headers))
	label_arry = [0]*len(headers)
	ind = headers.index(name)
	label_arry[ind] = 1
	return label_arry


def generate_validation_df():
	names = df.columns.tolist()
	drop_index = []
	names.insert(0, 'index')
	headers = generaete_catagories("catagory.csv")
	val_df = pd.DataFrame(columns=names)
	for head in headers:
		head_count = 0
		while head_count <= 5:
			for ind, row in df.iterrows():
				if row['CategoryName'] == head:
					val_df.loc[head_count] = df.loc[ind].tolist()
					drop_index.append(ind)
				else:
					continue
	df.drop(df.index[drop_index], inplace=True)
	return val_df

def create_featuresets(path):
	# path is the address to the folder containing images for training
	training_data = []
	for img in tqdm(os.listdir(path)):
		word_label = img.split('.')
		label = label_img(word_label)
		Path = os.path.join(path,img)
		im = scipy.misc.imread(Path, flatten=True)
		im = scipy.misc.imresize(im, size=(img_size,img_size))
		
		training_data.append([np.array(im),np.array(label)])

	#
	# np.save('train_data.npy', training_data)
	# print("Generated featureset")
	return training_data

def create_ValidSets(path):
	# path is the address to the folder containing images for training
	Valid_data = []
	for img in tqdm(os.listdir(path)):
		word_label = img.split('.')
		if word_label[1] in validate_index:
			label = label_img(img)
			Path = os.path.join(path,img)
			img = cv2.imread(Path,cv2.IMREAD_GRAYSCALE)

			Valid_data.append([np.array(img),np.array(label)])

	np.save('valid_data.npy', Valid_data)
	return Valid_data


if __name__ == "__main__":
	path = 'Images/Train/'
	train_data = create_featuresets(path)

	## IMAGE CLASSIFICATION
	conv_nn_inp = input_data(shape=[None, img_size, img_size, 1], name='image_input')

	conv_1 = conv_2d(conv_nn_inp, 32, 5, activation='relu')
	conv_1_pool = max_pool_2d(conv_1, 2)

	conv_2 = conv_2d(conv_1_pool, 32, 5, activation='relu')
	conv_2_pool = max_pool_2d(conv_2, 2)

	conv_3 = conv_2d(conv_2_pool, 64, 5, activation='relu')
	conv_3_pool = max_pool_2d(conv_3, 2)

	conv_4 = conv_2d(conv_3_pool, 128, 5, activation='relu')
	conv_4_pool = max_pool_2d(conv_4, 2)

	fc_layer_1 = fully_connected(conv_4_pool, 1024, activation='relu')
	fc_layer_1_drop = dropout(fc_layer_1, 0.85)

	fc_layer_2 = fully_connected(fc_layer_1_drop, 63, activation='softmax')
	Img_NN = regression(fc_layer_2, optimizer='adam', learning_rate = 1e-3, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(Img_NN, tensorboard_dir='log')

	# if os.path.exists('C/Users/sidha/OneDrive/DeepLearn/Stackline/{}'.format(MODEL_NAME)):
	# 	model.load(MODEL_NAME)
	# 	print('model loaded!')

	# validation_df = generate_validation_df()
	# validate_index = validation_df['index'].tolist()
	# try:

	# 	validation = np.load('valid_data.npy')
	# except:
	# 	train = create_featuresets("Images/Train/", validate_index)
	# 	validation = create_ValidSets("Images/Train/", validate_index)
	# #

	nos = random.sample(range(len(train_data)),250)
	train = [x for i,x in enumerate(train_data) if i not in nos]
	validation = [x for i,x in enumerate(train_data) if i in nos]
	X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
	Y = np.array([i[1] for i in train])

	test_x = np.array([i[0] for i in validation]).reshape(-1,img_size,img_size,1)
	test_y = [i[1] for i in validation]

	model.fit({'image_input': X}, {'targets': Y}, n_epoch=30, validation_set=({'image_input': test_x}, {'targets': test_y}),
	    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
	model.save(MODEL_NAME)
