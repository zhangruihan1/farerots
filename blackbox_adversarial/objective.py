import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from image2perturbed import GlassesPerturbation
from image2feature import get_feature, get_model

def cosine_similarity_single_system(im1, im2, backbone_type, batched_1 = True, batched_2 = True):
	if len(im1.shape) == 3:
		im1 = np.expand_dims(im1, axis=0)
		batched_1 = False
	if len(im2.shape) == 3:
		im2 = np.expand_dims(im2, axis=0)
		batched_2 = False
	images = np.concatenate((im1, im2), axis=0)

	m1 = get_model(backbone_type)
	features = get_feature(images, m1)

	f1 = features[:im1.shape[0]]
	f2 = features[im1.shape[0]:]

	return cosine_similarity(f1, f2)	



g1 = GlassesPerturbation()

from PIL import Image
i1 = np.array(Image.open('Screenshot 2022-06-28 213031.jpg').resize((112, 112)))
i2 = g1.perturb_single_image(np.random.normal(128, 128, size = (10, g1.n_var)), i1)

for i in range(10):
	im = Image.fromarray(i2[i])
	im.save(f"image_{i}.png")

csss = cosine_similarity_single_system(i1, i2, 'EfficientNet')
print(csss)