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

my_systems = ['FaceNet', 'r50', 'EfficientNet', 'ReXNet', 'AttentionNet', 'RepVGG', 'GhostNet', 'TF-NAS', 'LightCNN']


# box2 = WhiteboxWithMultipleSystems(*my_systems)


from PIL import Image

import csv

for my_system in ('r50', ):

	box = WhiteboxWithMultipleSystems(my_system)

	min_y_preds = []

	for k in trange(6000):

		# with open(f'y_preds_rex_{k}.csv', 'w', newline ='') as f:

			# write = csv.writer(f)
		if k%600 > 300:
			continue

		i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
		i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 

		i1_ = np.array(i2, copy=True)

		min_sim = 1

		for i in range(1000):
			g_1, _ = box.get_grads(i1_, i1)
			i1_ = fgsm_attack(i1_, 1, g_1)

			delta = np.clip(i2 - i1_, -4, 4)
			i1_ = np.clip(i2 + delta, 0, 255)

			sim_test = box.cosine_similarity(i1_, i1)

			if sim_test < min_sim:
				min_sim = sim_test
				Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_B_wblinf_{my_system}.png")
			if 0 in box.y_preds:
				print(box.y_preds)
				
			# write.writerow(box.y_preds)

		min_y_preds.append(min_sim)
		print('lowest similarity: ', min(min_y_preds))

	np.save('five' + '_min_y_preds', min_y_preds)

# save verfied judgements

# for my_system in my_systems:
# 	print(my_system)
# 	box = WhiteboxWithMultipleSystems(my_system)
# 	box2 = WhiteboxWithMultipleSystems(*my_systems)


# 	from PIL import Image

# 	import csv
	
# 	min_y_preds = []

# 	for k in trange(6000):

# 		if k%600 > 300:
# 			continue

# 		with open(f'y_preds_{my_system}_{k}.csv', 'w', newline ='') as f:

# 			write = csv.writer(f)

# 			i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
# 			i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 

# 			i1_ = np.array(i2, copy=True)

# 			min_sim = 1

# 			for i in range(1000):
# 				g_1, _ = box.get_grads(i1_, i1)
# 				i1_ = fgsm_attack(i1_, 1, g_1)

# 				delta = np.clip(i2 - i1_, -4, 4)
# 				i1_ = np.clip(i2 + delta, 0, 255)

# 				# sim_test = box2.cosine_similarity(i1_, i1)

# 				# if sim_test < min_sim:
# 				# 	min_sim = sim_test
# 				# 	Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_B_wblinf.png")
# 				write.writerow(box.y_preds)
# 				if 0 in box.y_preds:
# 					break
					

# 			min_y_preds.append(min_sim)
# 			print('lowest similarity: ', min(min_y_preds))

# 	np.save(my_system + '_min_y_preds', min_y_preds)

# save cosine similarity

# for my_system in my_systems:
# 	print(my_system)
# 	box = WhiteboxWithMultipleSystems(my_system)
# 	box2 = WhiteboxWithMultipleSystems(*my_systems)


# 	from PIL import Image

# 	import csv
	
# 	min_sims = []

# 	for k in trange(100):

# 		with open(f'sims_{my_system}_{k}.csv', 'w', newline ='') as f:

# 			write = csv.writer(f)

# 			i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
# 			i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 

# 			i1_ = np.array(i2, copy=True)

# 			min_sim = 1

# 			for i in range(100):
# 				g_1, _ = box.get_grads(i1_, i1)
# 				i1_ = fgsm_attack(i1_, 1, g_1)

# 				delta = np.clip(i2 - i1_, -3, 3)
# 				i1_ = np.clip(i2 + delta, 0, 255)

# 				sim_test = box2.cosine_similarity(i1_, i1)

# 				if sim_test < min_sim:
# 					min_sim = sim_test
# 					print(box2.sims)
# 					Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_B_wblinf.png")
					
# 				write.writerow(box2.sims)

# 			min_sims.append(min_sim)
# 			print('lowest similarity: ', min(min_sims))

# 	np.save(my_system + '_min_sims', min_sims)