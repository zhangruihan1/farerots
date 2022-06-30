import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from image2feature import get_feature, get_model

class BlackboxWithOneSystem:
	def __init__(self, backbone_type):
		super(BlackboxWithOneSystem, self).__init__()
		self.m1 = get_model(backbone_type)

	def cosine_similarity(self, im1, im2):
		# batched_1 = True
		# batched_2 = True
		if len(im1.shape) == 3:
			im1 = np.expand_dims(im1, axis=0)
			# batched_1 = False
		if len(im2.shape) == 3:
			im2 = np.expand_dims(im2, axis=0)
			# batched_2 = False

		images = np.concatenate((im1, im2), axis=0)		
		features = get_feature(images, self.m1)

		f1 = features[:im1.shape[0]]
		f2 = features[im1.shape[0]:]

		return cosine_similarity(f1, f2).squeeze(0)