import sys
sys.path.append(".")

import numpy as np

from blackbox_adversarial.image2perturbed import GlassesPerturbation
from algorithms import fgsm_attack

class WhiteboxGlassesPerturbation(GlassesPerturbation):
	def __init__(self):
		super(WhiteboxGlassesPerturbation, self).__init__()
		self.pname = 'wbglass'

	def perturb_single_image(self, i1, i1_, g_1, step_size = 2):
		mask = np.expand_dims(self.get_glasses_mask_for_image(i1.squeeze(0)).astype(bool), 0)
		i1_[mask] = fgsm_attack(i1_[mask], step_size, g_1[mask])
		i1_ = np.clip(i1_, 0, 255)
		return i1_

class LinfPerturbation:
	def __init__(self):
		super(LinfPerturbation, self).__init__()
		self.pname = 'wblinf'
	
	def perturb_single_image(self, i1, i1_, g_1, step_size = 1, max_step_size = 4):
		i1_ = fgsm_attack(i1_, step_size, g_1)
		delta = np.clip(i1 - i1_, -max_step_size, max_step_size)
		i1_ = np.clip(i1 + delta, 0, 255)
		return i1_