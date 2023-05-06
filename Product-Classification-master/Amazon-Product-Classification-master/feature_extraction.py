from PIL import Image
import numpy as np
import os

path = 'Images/Train/'
size = [128,128,3]
headers = ['Headphones\n', 'Cables\n', 'Security & Surveillance\n', 'Streaming Media\n', 'Television Accessories\n', 'Monitor Risers\n', 'Gaming Accessories\n', 'Video Games\n', 'Video Cameras\n', '3D Printers & Supplies\n', 'Drones\n', 'Mice\n', 'Computer Accessories\n', 'Keyboards\n', 'Monitor Mounts\n', 'Monitors\n', 'Office Electronics\n', 'Camera Accessories\n', 'Range Extenders\n', 'Ink & Toner\n', 'Car & Vehicle Electronics\n', 'Video Projectors\n', 'Tablet Accessories\n', 'Car Subwoofers & Amplifiers\n', 'Tablets\n', 'Laptop Accessories\n', 'Tripods & Monopods\n', 'Televisions\n', 'Batteries\n', 'Desktops\n', 'Laptops\n', 'Home Audio\n', 'GPS & Navigation\n', 'Radar Detectors\n', 'Mobile Phone Accessories\n', 'Headsets\n', 'Binoculars & Scopes\n', 'Modems\n', 'Cases & Screen Protectors\n', 'TV Mounts & Stands\n', 'eBook Readers & Accessories\n', 'Computer Data Storage\n', 'Portable Audio & Speakers\n', 'Power Management\n', 'Computer Components\n', 'Video Cards\n', 'Printers & Scanners\n', 'Memory Cards & Flash Drives\n', 'Unlocked Cell Phones\n', 'Wearable Technology\n', 'Motherboards\n', 'Telescopes\n', 'Routers & Networking\n', 'Car Dash Cams\n', 'Microphones and Accessories\n', 'Two Way Radios\n', 'Blu-ray and DVD Players\n', 'Standing Desks\n', 'Cameras\n', 'Switches\n', 'Calculators\n', 'Camera Lenses\n', 'Game Consoles\n']
headers = list(map(lambda s: s.strip(), headers))

def get_label(name):
	label = [0]*len(headers)
	ind = headers.index(name)
	label[ind] = 1
	return label

def change_pic(path):
	
	im = Image.open('Foto.jpg')
	im.save('Foto.png')

def get_feature(path):
	features = []
	for name in os.listdir(path):
		image_path = path + name
		
		image = Image.open(image_path)
		
		# image = image.convert("RGB")
		# im_ary = np.array(image)
		# im_arry = np.resize(im_ary, size)
		img_cat = name.split('.')
		img_name = img_cat[0] + '.' + img_cat[1]
		sav = '.png'
		save_path = path + img_name + sav
		image.save(save_path)
		# img_label = get_label(img_cat)
		# features.append([im_arry, np.array(img_label)])
	return True

if __name__ == "__main__":
	features = get_feature(path)
	# np.save('train_data.npy', features)
	# print (features[200])