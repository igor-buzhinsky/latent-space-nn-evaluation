''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange
from importlib import import_module
from typing import *

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

from .utils import load_weights, imsize_dict, nclass_dict, activation_dict, update_config_roots, seed_rng, name_from_config, count_parameters, prepare_z_y, sample as utils_sample, accumulate_standing_stats, sample_sheet, classes_per_sheet_dict, interp_sheet, prepare_parser, add_sample_parser

G, z_, y_, config, device = [None] * 5

def run():
    global G, z_, y_, config
    
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}
                
    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        load_weights(None, None, state_dict, config['weights_root'], 
                     config['experiment_name'], config['load_weights'], None,
                     strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
        if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
            config[item] = state_dict['config'][item]
  
    # update config (see train.py for explanation)
    config['resolution'] = imsize_dict[config['dataset']]
    config['n_classes'] = nclass_dict[config['dataset']]
    config['G_activation'] = activation_dict[config['G_nl']]
    config['D_activation'] = activation_dict[config['D_nl']]
    config = update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
  
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True
  
    # Import the model--this line allows us to dynamically select different files.
    model = import_module("." + config['model'], "biggan")
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else name_from_config(config))
  
    G = model.Generator(**config).to(device)
    count_parameters(G)
    G.eval()
  
    # Load weights
    #print('Loading weights...')
    # Here is where we deal with the ema--load ema weights or load normal weights
    load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)
    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size']) 
    z_, y_ = prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                         device=device, fp16=config['G_fp16'], 
                         z_var=config['z_var'])

def main(arg_device):
    global G, z_, y_, config, device
    device = arg_device
    # parse command line and run    
    parser = prepare_parser()
    parser = add_sample_parser(parser)
    config = vars(parser.parse_args())
    #print(config)
    run()

def create_sample(allowed_labels: List[int], builtin_decay: float) -> Tuple[torch.Tensor, torch.Tensor]:
    z_.sample_()
    y_ = torch.tensor([allowed_labels[torch.randint(len(allowed_labels), ())]], dtype=torch.long).to(device)
    G_z = G(z_ * (1 - builtin_decay), G.shared(y_))
    return G_z, y_
    
def decode_sample(latent_code: torch.Tensor, true_label: int, builtin_decay: float) -> torch.Tensor:
    y_ = torch.tensor([true_label], dtype=torch.long).to(device)
    G_z = G(latent_code * (1 - builtin_decay), G.shared(y_))
    return G_z
