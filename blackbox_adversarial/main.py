from image2perturbed import GlassesPerturbation
from objective import cosine_similarity_single_system
# from differential_evolution import differential_evolution
import numpy as np
from scipy.optimize._differentialevolution import differential_evolution


from PIL import Image
i1 = np.array(Image.open('Screenshot 2022-06-28 213031.jpg').resize((112, 112)))

perturb = GlassesPerturbation()
# X = np.random.normal(128, 128, size = (popsize, perturb.n_var))
# i2 = perturb.perturb_single_image(X, i1)
# csss = cosine_similarity_single_system(i1, i2, 'EfficientNet')


attack_result = differential_evolution(
	func=lambda xs: cosine_similarity_single_system(i1, perturb.perturb_single_image(xs, i1), 'EfficientNet'),
	bounds=[(0,255)] * perturb.n_var,
	maxiter=75,
	popsize=400,
	# recombination=1,
	# atol=-1,
	callback=lambda x: cosine_similarity_single_system(i1, perturb.perturb_single_image(np.expand_dims(x, axis=0), i1), 'EfficientNet') < 0.65,
	polish=False,
	# init=inits
)