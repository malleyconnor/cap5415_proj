import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import argparse
from torch.utils.tensorboard import SummaryWriter
from reconstructor import Reconstructor
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=32,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--mask_size',
                        type=int,
                        default=3,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='results',
                        help='Directory to put logging.')
    parser.add_argument('--data_path',
                        type=str,
                        default='./',
                        help='Path to input data')
    parser.add_argument('--split_factor',
                        type=int, 
                        default=1,
                        help='Splits images into split_factor^2 quadrants to expand dataset')
    parser.add_argument('--use_grayscale',
                        default=False,
                        action='store_true',
                        help='Does image reconstruction on grayscale images instead of RGB')
                       
                        
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    # Initialize the model and send to device 
    model = Reconstructor(FLAGS).to(device)
    num_params = sum(p.numel() for p in list(model.parameters()) if p.requires_grad)
    with open("%s/model_structure.txt" % (FLAGS.log_dir), "w") as structure_file:
        structure_file.write("Mask size: %d x %d x 3\n" % (FLAGS.mask_size, FLAGS.mask_size))
        structure_file.write("Training samples: %d\n" % len(model.xtrain))
        structure_file.write("Testing samples: %d\n" % len(model.xtest))
        structure_file.write("Model Parameters: %d\n" % (num_params))
        structure_file.write("Model layers: %s\n" % (str(model.layers)))
    model.train_test()
    model.generate_n_examples(5)
    