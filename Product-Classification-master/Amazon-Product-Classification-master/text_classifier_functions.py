import numpy as np
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from random import shuffle
#Token = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
lemmatizer = WordNetLemmatizer()
hm_lines = 100000
path = 'text.csv'

def label_text(label):
	headers = ['Headphones\n', 'Cables\n', 'Security & Surveillance\n', 'Streaming Media\n', 'Television Accessories\n', 'Monitor Risers\n', 'Gaming Accessories\n', 'Video Games\n', 'Video Cameras\n', '3D Printers & Supplies\n', 'Drones\n', 'Mice\n', 'Computer Accessories\n', 'Keyboards\n', 'Monitor Mounts\n', 'Monitors\n', 'Office Electronics\n', 'Camera Accessories\n', 'Range Extenders\n', 'Ink & Toner\n', 'Car & Vehicle Electronics\n', 'Video Projectors\n', 'Tablet Accessories\n', 'Car Subwoofers & Amplifiers\n', 'Tablets\n', 'Laptop Accessories\n', 'Tripods & Monopods\n', 'Televisions\n', 'Batteries\n', 'Desktops\n', 'Laptops\n', 'Home Audio\n', 'GPS & Navigation\n', 'Radar Detectors\n', 'Mobile Phone Accessories\n', 'Headsets\n', 'Binoculars & Scopes\n', 'Modems\n', 'Cases & Screen Protectors\n', 'TV Mounts & Stands\n', 'eBook Readers & Accessories\n', 'Computer Data Storage\n', 'Portable Audio & Speakers\n', 'Power Management\n', 'Computer Components\n', 'Video Cards\n', 'Printers & Scanners\n', 'Memory Cards & Flash Drives\n', 'Unlocked Cell Phones\n', 'Wearable Technology\n', 'Motherboards\n', 'Telescopes\n', 'Routers & Networking\n', 'Car Dash Cams\n', 'Microphones and Accessories\n', 'Two Way Radios\n', 'Blu-ray and DVD Players\n', 'Standing Desks\n', 'Cameras\n', 'Switches\n', 'Calculators\n', 'Camera Lenses\n', 'Game Consoles\n']
	label_arry = np.array(np.zeros(len(headers)))
	label_arry[headers.index(label)] = 1
	return label_arry

def create_lexicon(path):
	lexicon = []
	with open(path,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l.lower())
			lexicon += list(all_words)
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		#print(w_counts[w])
		if 500 > w_counts[w] > 20:
			l2.append(w)
	return l2

def sample_handling(sample,lexicon):
	featureset = []
	with open(sample,'r') as f1, open('classes_row_wise.csv', 'r') as f2:
		for l, labl in zip(f1, f2):
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			classification = label_text(labl)
			features = list(features)
			featureset.append([features,classification])
	shuffle(featureset)
	return featureset

def create_feature_sets_and_labels(path, test_size = 0.1):
	lexicon = create_lexicon(path)
	features = []
	features += sample_handling(path,lexicon)
	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels(path)
	# if you want to pickle this data:
	with open('Text_data.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
