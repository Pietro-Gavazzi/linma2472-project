#CODE PARAMETERS

# boolen: set at true if you want to save the model at each epoch
saveModelBool = True

#  Chose the dataset : "EMNIST_Letters" or "MNIST"
dataset = "MNIST"

# number of epochs 
num_epochs_gan = 7


# batch size
batch_size = 32


# dimension latent space for the generator
dim_latent_space = 100


# Adam optimization parameters
lr_gan = 0.0002
betas_gan = (0.5, 0.999)


# seed 
seed = 2472




#__________________________________________________________________________________________________________________________

# PyTorch is a common library for neural networks in Python. torch.nn is a module for building layers of Neural Networks.
import torch
from torch import nn
import torch.nn.functional as F

# TorchVision is part of the PyTorch environment. It is necessary to download the datasets MNIST (and EMNIST Letters)
import torchvision
import torchvision.transforms as transforms

# Usual mathematical stuff
import numpy as np
import math
# Plots
import matplotlib.pyplot as plt

# Timing
import time
from datetime import datetime
import time

_ = torch.manual_seed(seed) 


# Check if GPU is available

def chooseDevice():
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available via cuda")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU available")
    return device

device = chooseDevice()




    # Extracts the train and test sets from a chosen image dataset
# Values of pixels are normalized between -1 and 1

def getData(dataset = "MNIST", info = True):
    if dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=transform)
    elif dataset == "EMNIST_Letters":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.EMNIST(root=".", split="letters", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.EMNIST(root=".", split="letters", train=False, download=True, transform=transform)
    else:
        print("DATASET NOT CORRECTLY DEFINED")
    if info:
        print(train_set)
        print(test_set)
    return train_set, test_set



    # Choose which dataset to use

train_set, test_set = getData(dataset = dataset)


# Get information about the size of the train and test sets.

def dataSize(train_set, test_set):
    n_train_set = train_set.__len__()
    n_test_set = test_set.__len__()
    n_tot = n_train_set + n_test_set
    ratio_train_test = n_train_set / n_test_set
    percentage_train = n_train_set / n_tot
    percentage_test = 1.0 - percentage_train
    return n_train_set, n_test_set, n_tot, ratio_train_test, percentage_train, percentage_test



n_train_set, n_test_set, n_tot, ratio_train_test, percentage_train, percentage_test = dataSize(train_set, test_set)



# Divide dataset into batches

def divideInBatches(train_set, test_set, batch_size, n_train_set, n_test_set):
    # Train set
    n_batches_total = math.ceil(n_train_set/batch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Test set
    n_batches_total_test = math.ceil(n_test_set/batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, n_batches_total, n_batches_total_test


train_loader, test_loader, n_batches_total, n_batches_total_test = divideInBatches(train_set, test_set, batch_size, n_train_set, n_test_set)





#generator 

class Generator(nn.Module):

    def __init__(self, dim_latent_space):
        super().__init__()
        self.dim_latent_space = dim_latent_space
        self.fc = nn.Linear(self.dim_latent_space, 64*7*7)
        self.trans_conv1 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.trans_conv2 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.trans_conv3 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.trans_conv4 = nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        
    def forward(self, x):                   # Input = batch_size*dim_latent_space
        x = self.fc(x)                      # Output = batch_size*(64*7*7)
        x = x.view(-1, 64, 7, 7)            # Output = batch_size*64*7*7
        x = self.trans_conv1(x)             # Output = batch_size*64*14*14
        x = F.relu(self.batch_norm1(x))
        x = self.trans_conv2(x)             # Output = batch_size*32*14*14
        x = F.relu(self.batch_norm2(x))
        x = self.trans_conv3(x)             # Output = batch_size*16*14*14
        x = F.relu(self.batch_norm3(x))     
        x = self.trans_conv4(x)             # Output = batch_size*1*28*28
        x = torch.tanh(x)
        return x


generator = Generator(dim_latent_space).to(device=device)



class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1)     
        self.conv0_drop = nn.Dropout2d(0.25)
        self.conv1 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_drop = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.conv3_drop = nn.Dropout2d(0.25)
        self.fc = nn.Linear(128*7*7, 1)
    
    def forward(self, x):                               # Input = batch_size*1*28*28
        x = x.view(-1, 1, 28, 28)                       # Output = batch_size*1*28*28
        x = F.leaky_relu(self.conv0(x), 0.2)            # Output = batch_size*32*14*14
        x = self.conv0_drop(x)
        x = F.leaky_relu(self.conv1(x), 0.2)            # Output = batch_size*64*14*14
        x = self.conv1_drop(x)
        x = F.leaky_relu(self.conv2(x), 0.2)            # Output = batch_size*128*14*14
        x = self.conv2_drop(x)
        x = F.leaky_relu(self.conv3(x), 0.2)            # Output = batch_size*128*7*7
        x = self.conv3_drop(x)
        x = x.view(-1, 128*7*7)                         # Output = batch_size*(128*7*7)
        x = self.fc(x)                                  # Output = batch_size*1
        return x

discriminator = Discriminator().to(device=device)


def numberParameters(model, trainable = False, model_name = None):
    total_params = sum(param.numel() for param in model.parameters())
    if model_name != None:
        print("Number of parameters of " + model_name + ": " + str(total_params))
    return total_params



discriminator_params = numberParameters(discriminator, model_name = "discriminator")
generator_params = numberParameters(generator, model_name = "generator")


# Loss function for the GAN
#loss_function_gan = nn.BCELoss()
loss_function_gan = nn.BCEWithLogitsLoss()






optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_gan, betas = betas_gan)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr_gan, betas = betas_gan)

discriminator.parameters()



# Reference latent vectors

def generate_latent_vectors(dim_latent_space, batch_size):
    # generate latent vectors based on a standard normal distribution
    latent_vectors = torch.randn(dim_latent_space * batch_size).reshape(batch_size, dim_latent_space)

    return latent_vectors





# Training of the GAN
def train_gan(gen, dis, data_loader, dim_latent_space, saveModelBool, num_epochs_gan=num_epochs_gan, batch_size=batch_size):

    def saveModel(model, model_name):
        torch.save(model.state_dict(), './Models/' + model_name)
    

    iters = 0
    if (saveModelBool):
        disname = "dis0"
        genname = "gen0"
        saveModel(dis, disname)
        saveModel(gen, genname)

    for epoch in range(num_epochs_gan):
        print("\n begin training epoch " +str(epoch+1))
        for i, data in enumerate(data_loader, 0):
            """
            train the discriminator on the batch:
            """
            X_real = data[0].to(device)
            y_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            ## Train with all-real batch
            dis.zero_grad()
            # Forward pass real batch through D
            outputD_real = dis(X_real).view(-1)
            # compute optimizer's loss on real batch
            errD_real = loss_function_gan(outputD_real, y_real)
            # compute gradients for Discriminator in backward pass
            errD_real.backward()


            ## Train with all-fake batch
            # generate batch of latent vectors
            latent_vectors_fake = generate_latent_vectors(dim_latent_space, batch_size)
            # generate fake X, y (images, labels)
            X_fake = gen(latent_vectors_fake)
            y_fake = torch.zeros([batch_size])

            outputD_fake = dis(X_fake).view(-1)
            # compute optimizer's loss on fake batch
            errD_fake = loss_function_gan(outputD_fake, y_fake)
            errD_fake.backward()



            # update Discriminator
            optimizer_discriminator.step()

            """
            train the generator on the batch:
            """
            gen.zero_grad()
            X_fake = gen(latent_vectors_fake)
            y_fake = y_real # for the generator, labels are real for training the generator
            # perform another forward pass of all-fake batch using an updated Discriminator
            outputD_fake = dis(X_fake).view(-1)
            # compute generator's loss
            errG = loss_function_gan(outputD_fake, y_fake)
            errG.backward(retain_graph=True)

            # Update G
            optimizer_generator.step()
            iters += 1

            if (i%25==0):
                length = len(data_loader)
                print("done " +str(i) + "/" +str(length))

        print("end training epoch " +str(epoch+1) + "\n")

        if (saveModelBool):
            disname = "dis" +str(epoch+1)
            genname = "gen" +str(epoch+1)
            saveModel(dis, disname)
            saveModel(gen, genname)

    if (not saveModelBool):
        disname = "final_dis_" +str(epoch+1)+"_epochs"
        genname = "finale_gen_"+str(epoch+1)+"_epochs"
        saveModel(dis, disname)
        saveModel(gen, genname)
    return 



train_gan(generator, discriminator, train_loader, dim_latent_space,  saveModelBool, num_epochs_gan=num_epochs_gan, batch_size=batch_size)