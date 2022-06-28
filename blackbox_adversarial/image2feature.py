import sys
sys.path.append(".")
# sys.path.append("test_protocol")

from test_protocol.utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory

import torch

def get_model(backbone_type):
	backbone_factory = BackboneFactory(backbone_type, "test_protocol/backbone_conf.yaml")
	model_loader = ModelLoader(backbone_factory)
	return model_loader.load_model(f"checkpoints/{backbone_type}.pt")

get_model('EfficientNet')