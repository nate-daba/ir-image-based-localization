import os
import copy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.siamese_network import SiameseNet


def modify_model(model, args):
    
    if args.ground_color_space == 'L':
        model.ground_embedding.feature_extractor.extract_features.conv1_1 = \
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
        print('Ground embedding network modified for grayscale image')
    if args.aerial_color_space == 'L':
        model.aerial_embedding.feature_extractor.extract_features.conv1_1 = \
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
        print('Aerial embedding network modified for grayscale image')
    
    return model

def copy_weights(model, args):
    """Copy weights from src_net to dst_net. Layers with different dimensions are ignored."""
    
    # get grayscale ground network
    dst_net = model.module.ground_embedding.feature_extractor.extract_features 

    checkpoint = torch.load(args.ground_net_weights)

    # prepare pre-trained RGB ground network to copy from
    src_net = SiameseNet().cuda() 
    src_net = DDP(src_net, device_ids=[args.gpu])
    src_net.load_state_dict(checkpoint['state_dict'])
    src_net = src_net.module.ground_embedding.feature_extractor.extract_features

    # copy weights from pre-trained RGB ground net to grayscale ground net
    print("Copying weights from RGB ground net to grayscale ground net ...")
    print("="*115)
    for i, src_layer in enumerate(src_net):
        dst_layer = dst_net[i]
        if isinstance(src_layer, nn.Conv2d) and isinstance(dst_layer, nn.Conv2d):
            if src_layer.weight.shape == dst_layer.weight.shape:
                dst_layer.weight.data = copy.deepcopy(src_layer.weight.data)
                if src_layer.bias is not None and dst_layer.bias is not None:
                    dst_layer.bias.data = copy.deepcopy(src_layer.bias.data)
                print(f"Copied weights from {src_layer} to {dst_layer}")
            else:
                print('Conv dimensions do not match. Copying only matching dimensions.')
                dst_layer.weight.data = copy.deepcopy(src_layer.weight.data[:, 0:1, :, :])
                print(f"Copied weights from {src_layer} to {dst_layer}")
                if src_layer.bias is not None and dst_layer.bias is not None:
                    dst_layer.bias.data = copy.deepcopy(src_layer.bias.data)
        elif isinstance(src_layer, nn.BatchNorm2d) and isinstance(dst_layer, nn.BatchNorm2d):
            if src_layer.weight.shape == dst_layer.weight.shape:
                dst_layer.weight.data = copy.deepcopy(src_layer.weight.data)
                dst_layer.bias.data = copy.deepcopy(src_layer.bias.data)
                dst_layer.running_mean.data = copy.deepcopy(src_layer.running_mean.data)
                dst_layer.running_var.data = copy.deepcopy(src_layer.running_var.data)
                print(f"Copied weights from {src_layer} to {dst_layer}")
            else:
                print(f"Skipping {src_layer} and {dst_layer} because they have different dimensions")
        elif isinstance(src_layer, nn.Linear) and isinstance(dst_layer, nn.Linear):
            if src_layer.weight.shape == dst_layer.weight.shape:
                dst_layer.weight.data = copy.deepcopy(src_layer.weight.data)
                if src_layer.bias is not None and dst_layer.bias is not None:
                    dst_layer.bias.data = copy.deepcopy(src_layer.bias.data)
                print(f"Copied weights from {src_layer} to {dst_layer}")
            else:
                print(f"Skipping {src_layer} and {dst_layer} because they have different dimensions")
        else:
            print(f"Skipping {src_layer} and {dst_layer} because they are not the same type of layer")
    
    model.module.ground_embedding.feature_extractor.extract_features = dst_net
    print("Finished copying weights!")
    print("="*115)
    
