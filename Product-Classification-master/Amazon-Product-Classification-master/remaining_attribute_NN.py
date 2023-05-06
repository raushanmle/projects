import pandas as pd
from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()
import numpy as np
from random import shuffle
import tensorflow as tf



def gen_labels(path):
	headers = ['Headphones\n', 'Cables\n', 'Security & Surveillance\n', 'Streaming Media\n', 'Television Accessories\n', 'Monitor Risers\n', 'Gaming Accessories\n', 'Video Games\n', 'Video Cameras\n', '3D Printers & Supplies\n', 'Drones\n', 'Mice\n', 'Computer Accessories\n', 'Keyboards\n', 'Monitor Mounts\n', 'Monitors\n', 'Office Electronics\n', 'Camera Accessories\n', 'Range Extenders\n', 'Ink & Toner\n', 'Car & Vehicle Electronics\n', 'Video Projectors\n', 'Tablet Accessories\n', 'Car Subwoofers & Amplifiers\n', 'Tablets\n', 'Laptop Accessories\n', 'Tripods & Monopods\n', 'Televisions\n', 'Batteries\n', 'Desktops\n', 'Laptops\n', 'Home Audio\n', 'GPS & Navigation\n', 'Radar Detectors\n', 'Mobile Phone Accessories\n', 'Headsets\n', 'Binoculars & Scopes\n', 'Modems\n', 'Cases & Screen Protectors\n', 'TV Mounts & Stands\n', 'eBook Readers & Accessories\n', 'Computer Data Storage\n', 'Portable Audio & Speakers\n', 'Power Management\n', 'Computer Components\n', 'Video Cards\n', 'Printers & Scanners\n', 'Memory Cards & Flash Drives\n', 'Unlocked Cell Phones\n', 'Wearable Technology\n', 'Motherboards\n', 'Telescopes\n', 'Routers & Networking\n', 'Car Dash Cams\n', 'Microphones and Accessories\n', 'Two Way Radios\n', 'Blu-ray and DVD Players\n', 'Standing Desks\n', 'Cameras\n', 'Switches\n', 'Calculators\n', 'Camera Lenses\n', 'Game Consoles\n']
	header = list(map(lambda s: s.strip(), headers))
	label = []

	with open(path, 'r') as label_file:
		lines = label_file.readlines()
		print(len(lines))
		for line in lines:
			temp = [0]*len(header)
			ind = header.index(line.strip())
			temp[ind] = 1
			label.append(temp)
	return label

def get_features(path):
	labels = gen_labels('classes_row_wise.csv')
	features = []
	df = pd.read_csv(path)
	lb_results = lb_style.fit_transform(df["BrandName"])
	DF = pd.DataFrame(lb_results, columns=lb_style.classes_)

	feature_list = DF.values.tolist()
	for i in range(len(feature_list)):
		features.append([np.array(feature_list[i]), np.array(labels[i])])
	shuffle(features)
	return features

def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size

			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		saver.save(sess, "./3rd_NN_model")
		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

if __name__=="__main__":
	path = 'remaining_attr.csv'
	test_size = 0.1
	features = np.array(get_features(path))
	print(features)
	testing_size = int(test_size*len(features))
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	n_nodes_hl1 = 500
	n_nodes_hl2 = 1500
	n_nodes_hl3 = 500
	n_classes = 63
	batch_size = 200
	hm_epochs = 20

	x = tf.placeholder('float')
	y = tf.placeholder('float')

	hidden_1_layer = {'f_fum':n_nodes_hl1,
	                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
	                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'f_fum':n_nodes_hl2,
	                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'f_fum':n_nodes_hl3,
	                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'f_fum':None,
	                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
	                'bias':tf.Variable(tf.random_normal([n_classes])),}

	train_neural_network(x)
