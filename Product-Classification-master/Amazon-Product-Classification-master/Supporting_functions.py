import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
img_size = 225
df = pd.read_csv("training_data.csv", encoding = 'ISO-8859-1')

def generaete_catagories(cat_path):
	catagories = []
	with open('catagory.csv', 'r') as f:
		lines = f.readlines()
		for line in lines:
			catagories.append(line)
	print(catagories)

def label_img(img):
	word_label = img.split('.')[0]
	headers = ['Headphones\n', 'Cables\n', 'Security & Surveillance\n', 'Streaming Media\n', 'Television Accessories\n', 'Monitor Risers\n', 'Gaming Accessories\n', 'Video Games\n', 'Video Cameras\n', '3D Printers & Supplies\n', 'Drones\n', 'Mice\n', 'Computer Accessories\n', 'Keyboards\n', 'Monitor Mounts\n', 'Monitors\n', 'Office Electronics\n', 'Camera Accessories\n', 'Range Extenders\n', 'Ink & Toner\n', 'Car & Vehicle Electronics\n', 'Video Projectors\n', 'Tablet Accessories\n', 'Car Subwoofers & Amplifiers\n', 'Tablets\n', 'Laptop Accessories\n', 'Tripods & Monopods\n', 'Televisions\n', 'Batteries\n', 'Desktops\n', 'Laptops\n', 'Home Audio\n', 'GPS & Navigation\n', 'Radar Detectors\n', 'Mobile Phone Accessories\n', 'Headsets\n', 'Binoculars & Scopes\n', 'Modems\n', 'Cases & Screen Protectors\n', 'TV Mounts & Stands\n', 'eBook Readers & Accessories\n', 'Computer Data Storage\n', 'Portable Audio & Speakers\n', 'Power Management\n', 'Computer Components\n', 'Video Cards\n', 'Printers & Scanners\n', 'Memory Cards & Flash Drives\n', 'Unlocked Cell Phones\n', 'Wearable Technology\n', 'Motherboards\n', 'Telescopes\n', 'Routers & Networking\n', 'Car Dash Cams\n', 'Microphones and Accessories\n', 'Two Way Radios\n', 'Blu-ray and DVD Players\n', 'Standing Desks\n', 'Cameras\n', 'Switches\n', 'Calculators\n', 'Camera Lenses\n', 'Game Consoles\n']
	headers = list(map(lambda s: s.strip(), headers))
	label_arry = [0]*len(headers)
	label_arry[headers.index(word_label)] = 1
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
		print(word_label[1])
		label = label_img(img)
		Path = os.path.join(path,img)
		try:
			img1 = cv2.imread(Path,cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img1, (img_size,img_size))
			training_data.append([np.array(img),np.array(label)])
		except:
			pass

	np.save('train_data.npy', training_data)
	return training_data

def create_ValidSets(path, validate_index):
	# path is the address to the folder containing images for training
	Valid_data = []
	for img in tqdm(os.listdir(path)):
		word_label = img.split('.')
		if word_label[1] in validate_index:
			label = label_img(img)
			Path = os.path.join(path,img)
			img = cv2.imread(Path,cv2.IMREAD_UNCHANGED)
			img = cv2.resize(img, (img_size,img_size))
			Valid_data.append([np.array(img),np.array(label)])

	np.save('valid_data.npy', Valid_data)
	return Valid_data
	
def save_images(data, train=True):
	df = pd.read_csv(data, encoding = 'ISO-8859-1')

	if not train:
		DIR = "Images/Test/"
		for ind, row in df.iterrows():
			catagory = str(ind) + '.jpg'
			url = row['ImageUrl']
			path_to_save = os.path.join(DIR,catagory)
			URL.urlretrieve(url, path_to_save)
		return True
	else:
		DIR = "Images/Train/"
		for ind, row in df.iterrows():
			catagory = str(row['CategoryName']) + '.' + str(ind) + '.jpg'
			url = row['ImageUrl']
			path_to_save = os.path.join(DIR,catagory)
			URL.urlretrieve(url, path_to_save)
		return True


if __name__ == "__main__":

	train = create_featuresets("Images/Train/")
	print (train[2000])


