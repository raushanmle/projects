import pandas as pd
import os
import urllib.request as URL

def save_images(data, train=True):
	df = pd.read_csv(data, encoding = 'ISO-8859-1')
	if not train:
		DIR = "Images/Test/"
		for ind, row in df.iterrows():
			print(ind)
			if ind > 21912:
				catagory = str(ind) + '.jpg'
				url = row['ImageUrl']
				path_to_save = os.path.join(DIR,catagory)
				if not path_to_save in os.listdir(DIR):
					try:
						URL.urlretrieve(url, path_to_save)
					except:
						print(url,'\n')
				else:
					print(str(ind))
					continue
		return True
	else:
		DIR = "Images/Train/"
		for ind, row in df.iterrows():
			catagory = str(row['CategoryName']) + '.' + str(ind) + '.jpg'
			url = row['ImageUrl']
			path_to_save = os.path.join(DIR,catagory)
			URL.urlretrieve(url, path_to_save)
		return True


save_images('test_data.csv', train=False)
