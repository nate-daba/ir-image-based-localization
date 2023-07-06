from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import FeatureExtractor


Tensor = torch.Tensor

class SiameseNet(nn.Module):
    """Defines Siamese network structure made up of two identical feature exractors.
    Args:
        ground_net (nn.Module): feature extractor for the ground view images.
        aerial_net (nn.Module): feature extractor for the aerial view images.
    
    """
    def __init__(self)->None:
        super(SiameseNet, self).__init__()
        
        print('Creating SiameseNet model ...')
        # create feature extractor for ground images 
        self.ground_embedding = nn.Sequential(OrderedDict([
            ('feature_extractor', FeatureExtractor(initialize_weights=True, extractor_type='ground')), 
            ('fc', nn.Linear(512, 1024, bias=True))]))
        # create feature extractor for aerial images 
        self.aerial_embedding = nn.Sequential(OrderedDict([
            ('feature_extractor', FeatureExtractor(initialize_weights=True, extractor_type='aerial')), 
            ('fc', nn.Linear(512, 1024, bias=True))]))
        # share weights of fc layer between feature extractors
        self.ground_embedding.fc = self.aerial_embedding.fc
        
    def forward(self, 
                ground_image: Tensor, 
                aerial_image: Tensor)->Tuple[Tensor]:
        """Extracts features from ground and aerial view images. Passes features through 
        linear layer. Computes L2 norms of linear layer outputs to compute final embedding.
        
        Args:
            ground_image (Tensor): ground view image.
            aerial_image (Tensor): aerial view image.
        Returns:
            aerial_global (Tensor): L2 normalized embedding of aerial image.
            ground_global (Tensor): L2 normalized embedding of ground image.
            aerial_features (Tensor): extracted features of aerial image.
            ground_features (Tensor): extracted features of ground image.
            aerial_fc (Tensor): output of linear layer for aerial image.
            ground_fc (Tensor): output of linear layer for ground image.
            
        """
        # fully connected (fc) layer output: (B, 1024)
        ground_embedding = self.ground_embedding(ground_image)
        aerial_mebedding = self.aerial_embedding(aerial_image)
        # L2 normalize to get final embedding: (B, 1024)
        ground_embedding = F.normalize(ground_embedding, p=2, dim=1)
        aerial_mebedding = F.normalize(aerial_mebedding, p=2, dim=1)
        
        return ground_embedding, aerial_mebedding