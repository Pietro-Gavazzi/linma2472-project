
#  Chose the dataset : "EMNIST_Letters" or "MNIST"
dataset = "MNIST"


# number of epochs 
num_epochs_cnn = 5

# batch size
batch_size = 32



# Adam optimization parameters
lr_cnn = 0.0002
betas_cnn = (0.5, 0.999)



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



def numberParameters(model, trainable = False, model_name = None):
    total_params = sum(param.numel() for param in model.parameters())
    if model_name != None:
        print("Number of parameters of " + model_name + ": " + str(total_params))
    return total_params




def numberClasses(dataset):
    n_classes = len(dataset.classes)
    return n_classes


n_classes = numberClasses(train_set)



class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Sequential(             # Images input is batch_size*1*28*28
            nn.Conv2d(1, 16, 5, 1, 2),          # Output is batch_size*16*28*28
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),        # Output is batch_size*16*14*14
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),         # Output is batch_size*32*14*14
            nn.ReLU(),                      
            nn.MaxPool2d(2),                    # Output is batch_size*32*7*7
        )
        # fully connected layer, output one number for each class
        self.out = nn.Sequential(         
            nn.Linear(32 * 7 * 7, self.n_classes),          # Output is batch_size*n_classes
        )
    
    def scaleToProbabilities(self, outmap, scale_factor = 3): 
        # scale_factor allows to scale your output before passing it into the softmax function 
        # in order to get numbers interpratble as probabilities
        
        flattened_outmap = outmap.view(outmap.shape[0], -1)
        outmap_std = torch.std(flattened_outmap, dim = 1).view(-1, 1)
        outmap_scaled_std = torch.div(outmap, outmap_std)
        probabilities = nn.functional.softmax(outmap_scaled_std*scale_factor, dim=1)
        return probabilities
    
    def predictLabels(self, outmap, scale_factor = 3):
        probabilities = self.scaleToProbabilities(outmap, scale_factor = 3)
        certainty, predicted_labels = torch.max(probabilities, 1)
        predicted_labels = predicted_labels.data.squeeze()
        certainty = certainty.data.squeeze()
        return predicted_labels, probabilities, certainty
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN(n_classes).to(device=device)
cnn_params = numberParameters(cnn, model_name = "classifier")


loss_function_cnn = nn.CrossEntropyLoss()



optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=lr_cnn, betas = betas_cnn)


def computeCNNAccuracy(y_predicted, y_true):
    accuracy = torch.sum((y_predicted == y_true) / len(y_true))
    return accuracy



# Training of the CNN
def train_CNN(classifier, num_epochs_cnn, dataloader):


    def saveModel(model, model_name):
        torch.save(model.state_dict(), './Models/' + model_name)
    

    iters = 0
    classifier.train()
    for epoch in range(num_epochs_cnn):
        print("\n begin training epoch " +str(epoch+1))

        for i, data in enumerate(dataloader,0):
            #train the cnn classifier on each batch i --> i suppose it's the real batch?
            xdata = data[0].to(device)
            ydata = data[1].to(device)

            #Settting the gradients of all optimized torch.Tensors to zero.
            classifier.zero_grad()
            #training the cnn classifer wrt the real x data
            result = classifier(xdata)
            #returning a new tensor with the same data as the self tensor but of a different shape
            #computing loss
            loss_cnn = loss_function_cnn(result, ydata)

            #backpropagate the error in cnn
            loss_cnn.backward()

            optimizer_cnn.step()

            y_hat = classifier.predictLabels(result)[0]
            acc_cnn = computeCNNAccuracy(y_hat, ydata)
            # print some info evey 50 iters
            if i % 25 == 0:
                print("done %i/%i : Acc=%.4f" %(i, len(dataloader), acc_cnn))
            iters += 1
        print("end training epoch " +str(epoch+1) + "\n")

    classifier.eval()

    saveModel(cnn, "cnn_"+str(epoch+1)+"_epochs")
    return 


train_CNN(cnn, num_epochs_cnn, train_loader)