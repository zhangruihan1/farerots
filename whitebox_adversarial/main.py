# from image2perturbed import GlassesPerturbation
from objective import WhiteboxWithOneSystem, WhiteboxWithMultipleSystems
import numpy as np
# from scipy.optimize._differentialevolution import differential_evolution
# from differential_evolution import differential_evolution
from algorithms import fgsm_attack


from PIL import Image
from tqdm import trange
# i1 = np.expand_dims(np.array(Image.open('mb1.jpg').resize((112, 112))), 0)
# i2 = np.expand_dims(np.array(Image.open('mb2.jpg').resize((112, 112))), 0)
# i1 = np.expand_dims(np.array(Image.open('Screenshot 2022-06-28 213031.jpg').resize((112, 112))), 0)

# perturb = GlassesPerturbation()


box = WhiteboxWithMultipleSystems('ReXNet', 'GhostNet')
box2 = WhiteboxWithMultipleSystems('EfficientNet')


from PIL import Image


min_sims = []

for k in trange(6000):

	i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
	i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 

	i1_ = np.array(i2, copy=True)

	min_sim = 1

	# good_examples = []


	for i in range(1000):
		g_1, _ = box.get_grads(i1_, i1)
		i1_ = fgsm_attack(i1_, 1, g_1)

		delta = np.clip(i2 - i1_, -3, 3)
		i1_ = np.clip(i2 + delta, 0, 255)

		sim_test = box2.cosine_similarity(i1_, i1)

		if sim_test < min_sim:
			min_sim = sim_test
			print(box.cosine_similarity(i1_, i1), min_sim)
			Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_B_wblinf.png")
			
			# if min_sim < 0.4:
			# 	print(k)
			# 	good_examples.append(k)

	min_sims.append(min_sim)

np.save(min_sims, 'min_sims')