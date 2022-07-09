import sys
sys.path.append(".")
sys.path.append('..')

from test_protocol.utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory

import torch
import torch.nn.functional as F

import numpy as np

from insightface.recognition.arcface_torch.backbones import get_model as insf_get_model
from facenet_pytorch import InceptionResnetV1

def get_model(backbone_type):

	if backbone_type in ('r18', 'r34', 'r50', 'r100'):
		m = insf_get_model(backbone_type).cuda()
		m.load_state_dict(torch.load("checkpoints/" + backbone_type + ".pt"))
		return m.eval()

	elif backbone_type == "FaceNet":
		return InceptionResnetV1(pretrained='vggface2').eval().cuda()

	else:	
		backbone_factory = BackboneFactory(backbone_type, "test_protocol/backbone_conf.yaml")
		model_loader = ModelLoader(backbone_factory)
		return model_loader.load_model("checkpoints/" + backbone_type + ".pt").eval()

def get_feature(image, m, batched = True, mini_batch_size = 256): # with mini-batch
	if len(image.shape) == 3:
		image = np.expand_dims(image, axis=0)
		batched = False
	image = (image.transpose((0, 3, 1, 2)) - 127.5) / 128.0
	image = torch.from_numpy(image.astype(np.float32)).contiguous()
	
	features = []
	with torch.no_grad():
		for i in range(int(np.ceil(len(image) / mini_batch_size))):
			image_ = image[mini_batch_size * i:mini_batch_size * (i + 1)].to(next(m.parameters()).device)
			feature_ = m(image_)
			feature_ = F.normalize(feature_)
			features.append(feature_.cpu().numpy())

	feature = np.concatenate(features, axis=0)
		
	if not batched:
		feature = feature[0]

	return feature

# thresholds = {
# 	'FaceNet':,
# 	'r50', 'EfficientNet', 'ReXNet', 'AttentionNet', 'RepVGG', 'GhostNet', 'TF-NAS', 'LightCNN'}	