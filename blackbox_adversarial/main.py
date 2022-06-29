from image2perturbed import GlassesPerturbation
from objective import cosine_similarity_single_system
import numpy as np
from scipy.optimize._differentialevolution import differential_evolution
# from differential_evolution import differential_evolution


from PIL import Image
i1 = np.array(Image.open('Screenshot 2022-06-28 213031.jpg').resize((112, 112)))

perturb = GlassesPerturbation()


attack_result = differential_evolution(
	func=lambda xs: cosine_similarity_single_system(i1, perturb.perturb_single_image(xs, i1), 'EfficientNet'),
	bounds=[(0,255)] * perturb.n_var,
	maxiter=75,
	popsize=1,
	# recombination=1,
	# atol=-1,
	callback=lambda x: cosine_similarity_single_system(i1, perturb.perturb_single_image(xs, i1), 'EfficientNet') < 0.65,
	polish=False,
	# init=inits,
	disp=True,
	# workers=2,
)