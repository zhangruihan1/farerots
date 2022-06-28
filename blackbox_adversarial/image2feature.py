import sys
sys.path.append(".")

from test_protocol.utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory

import torch
import torch.nn.functional as F

import numpy as np

def get_model(backbone_type):
	backbone_factory = BackboneFactory(backbone_type, "test_protocol/backbone_conf.yaml")
	model_loader = ModelLoader(backbone_factory)
	return model_loader.load_model(f"checkpoints/{backbone_type}.pt").eval()

def get_feature(image, m, batched = True):
	if len(image.shape) == 3:
		image = np.expand_dims(image, axis=0)
		batched = False
	image = (image.transpose((0, 3, 1, 2)) - 127.5) / 128.0
	image = torch.from_numpy(image.astype(np.float32)).contiguous()
	

	with torch.no_grad(): 
		images = image.to(next(m.parameters()).device)
		feature = m(images)
		feature = F.normalize(feature)
		feature = feature.cpu().numpy()
		
	if not batched:
		feature = feature[0]

	return feature