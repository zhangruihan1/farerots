import mxnet as mx
import os
from PIL import Image
import pickle
import numpy as np


import pandas as pd
import torch
import torchvision

class GeneralDataset(torch.utils.data.Dataset):
	def __init__(self, source, pname):
		super(GeneralDataset, self).__init__()
		# self.persons = get_person_id_category(self.record)
		self.samples = os.listdir(source)
		perturbations = ['wbglass', 'wbl2', 'wblinf', 'wbrect']
		new_samples = []
		final_perturb = {}
		for sample in self.samples:
			if '.jpg' in sample:
				new_samples.append(sample)
			else:
				for pname in perturbations:
					if pname in sample:
						temp = sample.split('.')[0].split('_')
						img_ind = int(temp[1])
						try:
							pert_ind = int(temp[-1])
						except:
							continue
						if (pname,img_ind) in final_perturb:
							if pert_ind > final_perturb[pname,img_ind][0]:
								final_perturb[pname,img_ind] = (pert_ind, sample)
						else:
							final_perturb[pname,img_ind] = (pert_ind, sample)
		temp_samples = [y for (x, y) in final_perturb.values()]
		new_samples = new_samples + temp_samples
		print(len(new_samples))
		self.samples = new_samples
		self.pname = pname
		self.source = source

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		img = (torchvision.io.read_image(self.source + '/' + self.samples[index], mode=torchvision.io.ImageReadMode.RGB) - 127.5) / 128.0
		return torchvision.transforms.Resize((112, 112))(img)


lfw_set_ = GeneralDataset('../lfw', 'lfw_perturb')
# ba_lfw_set = torch.utils.data.DataLoader(ba_lfw_set_, batch_size = 64, shuffle = False)

from blackbox_adversarial.image2feature import get_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm import tqdm

my_systems = ['FaceNet', 'r50', 'EfficientNet', 'ReXNet', 'AttentionNet', 'RepVGG', 'GhostNet', 'TF-NAS', 'LightCNN']
all_data = {}

for dset_ in [lfw_set_, ]:
	dset = torch.utils.data.DataLoader(dset_, batch_size = 256, shuffle = False)
	for my_system in my_systems:
		model = get_model(my_system)['m']

		features_ = []
		for batch in tqdm(dset):
			with torch.no_grad():
				feat = model(batch.to(device))
				features_.append(feat.detach().cpu())

		features = torch.cat(features_).numpy()
		all_data[dset_.pname, my_system] = dict(zip(dset_.samples, features))


		with open('erp_data_C1C2.pkl', 'wb') as f:
			pickle.dump(all_data, f)