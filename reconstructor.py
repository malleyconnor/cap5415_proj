from re import I
import torch.nn as nn
import torch
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
import csv

class Reconstructor(nn.Module):
    def __init__(self, FLAGS):
        super(Reconstructor, self).__init__()
        self.FLAGS = FLAGS
        random.seed()

        self.num_epochs = FLAGS.num_epochs
        self.batch_size = FLAGS.batch_size
        self.lr = FLAGS.lr
        self.mask_size  = FLAGS.mask_size

        # Select input image resolution
        self.res = (32, 32)
        self.res = (int(self.res[0] / self.FLAGS.split_factor), int(self.res[1] / self.FLAGS.split_factor))

        if FLAGS.use_grayscale:
            self.input_channels = 1
        
        # Check if cuda is available
        use_cuda = torch.cuda.is_available()
        
        # Set proper device based on cuda availability 
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("Torch device selected: ", self.device)

        # Reading input images
        #this_dir = os.listdir(FLAGS.data_path)
        #im_paths = []
        #for im in this_dir:
        #    im_paths.append(FLAGS.data_path + '/' + im)
        #print("%d input images detected..." % (len(im_paths)))

        #nsamples = 16384
        #if self.FLAGS.split_factor != 1:
        #    nsamples = nsamples * (self.FLAGS.split_factor**2)
        #    data = torch.zeros((nsamples, self.input_channels, self.res[0], self.res[1]), dtype=torch.float32)
        #else:
        #    data = torch.zeros((nsamples, self.input_channels, self.res[0], self.res[1]), dtype=torch.float32)

        #print("Loading input images...")
        #if (self.FLAGS.split_factor != 1):
        #    for i in tqdm(range(int(nsamples/(self.FLAGS.split_factor**2)))):
        #        if self.input_channels == 1:
        #            im = torch.tensor(cv2.imread(im_paths[i], 0), dtype=torch.float32)
        #            im = im.view(self.input_channels, im.shape[0], im.shape[1])
        #        else:
        #            im = torch.tensor(np.transpose(cv2.imread(im_paths[i]), axes=[2,0,1]), dtype=torch.float32)
        #        
        #        
        #        im = self.pad_image_bottom_right(im, res=(self.res[0]*self.FLAGS.split_factor, self.res[1]*self.FLAGS.split_factor))
        #        j = int(i*(self.FLAGS.split_factor**2))
        #        for k in range(self.FLAGS.split_factor):
        #            x_ind = k * self.res[0]
        #            for l in range(self.FLAGS.split_factor):
        #                y_ind = l * self.res[1]
        #                data[j] = im[:, max(0, x_ind):x_ind+self.res[0], max(0, y_ind):y_ind+self.res[1]]
        #                j += 1
        #else:
        #    for i in tqdm(range(nsamples)):
        #        data[i] = self.pad_image_bottom_right(torch.tensor(np.transpose(cv2.imread(im_paths[i]), axes=[2,0,1]), dtype=torch.float32))
        #self.xtrain, self.xtest = train_test_split(data, test_size=0.2, shuffle=True)

        transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor()
        ])
        self.xtrain = CIFAR10(FLAGS.data_path, train=True, transform=transform, download=True)
        self.xtest = CIFAR10(FLAGS.data_path, train=True, transform=transform, download=True)
        self.input_channels=3
        # Splits data between train/test
        # Labels will be dynamic from masking
        self.train_loader = DataLoader(self.xtrain, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.xtest, batch_size=self.batch_size)

        # Defining model layers
        self.out_channels = 16
        self.conv1 = nn.Conv2d(in_channels=self.input_channels,out_channels=self.out_channels, kernel_size=5, stride=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, kernel_size=3, stride=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels, kernel_size=3, stride=1, padding_mode='zeros')
        self.mp1   =    nn.MaxPool2d(kernel_size=3)
        self.mp2  =  nn.MaxPool2d(kernel_size=3)
        self.mp3  =  nn.MaxPool2d(kernel_size=3)
        self.flatten =   nn.Flatten(start_dim=1)
        self.lin1 =   nn.Linear(in_features=1024, out_features=256)
        self.lin2 =   nn.Linear(in_features=256, out_features=256)
        self.lin3 =   nn.Linear(in_features=256, out_features=256)
        self.cls =   nn.Linear(in_features=256, out_features=self.input_channels*(self.mask_size**2))
        self.drop = nn.Dropout(0.25)
        self.relu = nn.Sigmoid()
        self.layers = nn.ModuleList([self.conv1, self.drop, nn.Tanh(), self.conv2, self.drop, nn.Tanh(), self.conv3, nn.Tanh(), self.drop, self.mp1, 
                                     self.flatten, self.lin1, self.drop, nn.Sigmoid(), self.lin2, self.drop, nn.Sigmoid(), self.lin3, self.drop, nn.Sigmoid(), self.cls, nn.Sigmoid()])


        self.criterion = torch.nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) 


    def pad_image_bottom_right(self, im, res=-1):
        im_shape = np.shape(im)
        if res == -1:
            res = self.res
        if im_shape[1] == res[0] and im_shape[2] == res[1]:
            return im

        new_im = torch.zeros((self.input_channels, res[0], res[1]), dtype=torch.float32)
        new_im[:, 0:im_shape[0], 0:im_shape[1]] = im[:, 0:min(res[0], im_shape[0]), 0:min(res[1], im_shape[1])]

        return new_im


    def forward(self, X):
        # Iterate over batch and perform masking
        labels = torch.zeros((len(X),self.input_channels, self.mask_size,self.mask_size), dtype=torch.float32).cuda()
        masked_data = torch.clone(X).cuda()
        #X = X.to(self.device)
        
        # masked_data = torch.transpose(torch.clone(X), 2, 3).to(self.device)
        #masked_data = torch.transpose(masked_data, 1, 2).to(self.device)
        mask_x = int(self.res[0]/2) - int(self.mask_size/2)
        mask_y = int(self.res[1]/2) - int(self.mask_size/2)
        for i in range(len(X)):
            labels[i] = X[i, :, mask_x:mask_x+self.mask_size, mask_y:mask_y+self.mask_size].detach().clone()
            masked_data[i] = torch.div(masked_data[i], 255)
            masked_data[i, :, mask_x:mask_x+self.mask_size, mask_y:mask_y+self.mask_size] = torch.ones((self.input_channels, self.mask_size, self.mask_size), dtype=torch.float32) * -1#-1


        # Sending thru model
        for layer in self.layers:
            masked_data = layer(masked_data)


        outputs = masked_data.view(len(X), self.input_channels, self.mask_size, self.mask_size)
        return outputs, labels


    # Generates a random set of example test images to be used for the project report
    def generate_n_examples(self, n):
        try:
            os.mkdir('example_results')
        except OSError:
            ...


        random.seed()
        im_inds = random.sample(list(np.linspace(0, len(self.xtest)-1, len(self.xtest), endpoint=True, dtype=int)), n)
        #masked_ims = torch.zeros(len(im_inds), self.input_channels, self.res[0], self.res[1], dtype=torch.float32).cuda()
        masked_ims = torch.zeros(len(im_inds), self.input_channels, self.res[0], self.res[1], dtype=torch.float32)
        #x_pos = []
        #y_pos = []
        mask_x = int(self.res[0]/2) - int(self.mask_size/2)
        mask_y = int(self.res[1]/2) - int(self.mask_size/2)
        for i in range(len(im_inds)):
        #    mask_x = random.randint(0, self.res[0]-self.mask_size)
        #    mask_y = random.randint(0, self.res[1]-self.mask_size)
            masked_ims[i] = self.xtest[im_inds[i]][0]

        self.eval()
        with torch.no_grad():
            predictions, labels = self(masked_ims)
        #predictions = torch.round(torch.mul(predictions, 255)).type(torch.uint8)
        #predictions = np.transpose(predictions.cpu().numpy(), axes=[0, 2, 3, 1])

        for i in range(len(im_inds)):
            masked_ims[i, :, mask_x:mask_x+self.mask_size, mask_y:mask_y+self.mask_size] = predictions[i] 

            # Grayscale
            if self.input_channels == 1:
                plt.imsave("example_results/testim_%d.png" % (i), torch.squeeze(torch.round(torch.mul(masked_ims[i], 255)).type(torch.uint8)).cpu().numpy(), cmap="gray")
            # RGB
            else:
                plt.imsave("example_results/testim_%d.png" % (i), np.transpose(torch.round(torch.mul(masked_ims[i], 255)).type(torch.uint8).cpu().numpy(), axes=[1, 2, 0]))



    def train_test(self):
        self.train_losses = []
        self.test_losses = []

        # Logging train/test accuracies and losses
        # output
        if not os.path.exists(self.FLAGS.log_dir):
            os.makedirs(self.FLAGS.log_dir)
        train_loss_file = open('%s/train_loss.csv' % (self.FLAGS.log_dir), 'w')
        test_loss_file  = open('%s/test_loss.csv' % (self.FLAGS.log_dir), 'w')
        train_loss_writer = csv.writer(train_loss_file)
        test_loss_writer  = csv.writer(test_loss_file)

        for epoch in range(self.num_epochs):
            print("EPOCH %d" % (epoch))

            # Training
            self.train()
            losses = []
            for batch_idx, data in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                data, _ = data
                data = data.cuda()
                outputs, labels = self(data)
                #outputs = torch.mul(outputs, 255)
                loss = self.criterion(outputs, labels)

                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()

            self.train_losses.append(np.mean(losses))

            # Testing
            self.eval()
            losses = []
            with torch.no_grad():
                for batch_idx, data in enumerate(self.test_loader):
                    data, _ = data
                    data = data.cuda()
                    outputs, labels = self(data)
                    #outputs = torch.mul(outputs, 255)
                    loss = self.criterion(outputs, labels)
                    losses.append(loss.item())
            
            self.test_losses.append(np.mean(losses))
        
            print("\tTrain loss: %.3f " % (self.train_losses[epoch]))
            print("\tTest loss: %.3f " % (self.test_losses[epoch]))
            train_loss_writer.writerow([self.train_losses[epoch]])
            test_loss_writer.writerow([self.test_losses[epoch]])

        train_loss_file.close() 
        test_loss_file.close() 
        plt.plot(np.linspace(1,self.num_epochs, self.num_epochs, endpoint=True), self.train_losses, label="Train")
        plt.plot(np.linspace(1,self.num_epochs, self.num_epochs, endpoint=True), self.test_losses, label="Test")
        plt.title("Train/Test Loss vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss (%d x %d x 3)" % (self.mask_size, self.mask_size))
        plt.legend()
        plt.savefig("%s/loss.png" % (self.FLAGS.log_dir))
        

