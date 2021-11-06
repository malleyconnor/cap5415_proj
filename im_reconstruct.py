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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()


    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)


    # Initialize the model and send to device 
    model = Reconstructor(FLAGS).to(device)
    