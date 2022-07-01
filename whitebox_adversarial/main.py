# from image2perturbed import GlassesPerturbation
from objective import WhiteboxWithOneSystem, WhiteboxWithMultipleSystems
import numpy as np
# from scipy.optimize._differentialevolution import differential_evolution
# from differential_evolution import differential_evolution
from algorithms import fgsm_attack


from PIL import Image
# i1 = np.expand_dims(np.array(Image.open('mb1.jpg').resize((112, 112))), 0)
# i2 = np.expand_dims(np.array(Image.open('mb2.jpg').resize((112, 112))), 0)
i1 = np.expand_dims(np.array(Image.open('Screenshot 2022-06-28 213031.jpg').resize((112, 112))), 0)

# perturb = GlassesPerturbation()


box = WhiteboxWithMultipleSystems('ReXNet', 'GhostNet')
box2 = WhiteboxWithMultipleSystems('EfficientNet')


from PIL import Image

i1_ = np.array(i1, copy=True)
for i in range(1000):
	g_1, _ = box.get_grads(i1_, i1)
	i1_ = fgsm_attack(i1_, 1, g_1)

	delta = np.clip(i1 - i1_, -3, 3)
	i1_ = np.clip(i1 + delta, 0, 255)

	Image.fromarray(i1.squeeze(0).astype(np.uint8)).save("original.png")
	Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"perturbed_{i}.png")

	print(i, box.cosine_similarity(i1_, i1), box2.cosine_similarity(i1_, i1))