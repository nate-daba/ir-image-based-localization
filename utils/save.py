import os
import shutil

import torch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):
    filename = filename if args.ground_color_space=='RGB' else \
    'grayscale_expt_' + filename
    torch.save(state, os.path.join(args.save_path, filename))
    if is_best:
        best_model_filename = 'model_best.pth.tar' if \
        args.ground_color_space=='RGB' else \
        'grayscale_expt_model_best.pth.tar'
        
        shutil.copyfile(os.path.join(args.save_path, filename), 
                        os.path.join(args.save_path, best_model_filename))

def save_image(image, image_path):
    pass
