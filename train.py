# Import here
from functions_col import import_func, process_image, imshow, predict, save_checkpoint, load_checkpoint, load_train_data, load_val_data, train_model
import argparse

import pandas as pd
import numpy as np

import torch
from torch import nn,optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session

from PIL import Image

# Main 
if __name__ == '__main__':
    
    # Set up ArgumentParser to retrieve data from command line
    parser = argparse.ArgumentParser(
        description="AI Image Classfication App"
    )
    
    # Add parameters
    parser.add_argument('data_dir', help="Data Directory of Images for training")
    parser.add_argument('--output_features', help="Number of classese need to classify", default=102, type=int)
    parser.add_argument('--save_dir', help="Saved Directory of trained model", default='checkpointTrainedModel.pth')
    parser.add_argument('--arch', help="Pretrained Torchvision Neural Network model architecture", default='vgg11')
    parser.add_argument('--learning_rate', help="Learning Rate for classifier", default=0.0008, type=float)
    parser.add_argument('--hidden_units', help="Number of hidden units of classifier", default=512, type=int)
    parser.add_argument('--epochs', help="Number of time to train model", default=1, type=int)
    parser.add_argument('--gpu', help="Train on GPU", action='store_true')
    parser.add_argument('--trained', help="Train on old model", action='store_true')
    
    
    #Parser the arguments
    args = parser.parse_args()
    
    # get values parsed from command line
    data_dir = args.data_dir
    output_features = args.output_features
    model_arch = args.arch
    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    save_dir = args.save_dir
    trained = args.trained
    
    
    # set train, valid directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # load train, valid data
    trainloader, train_data = load_train_data(train_dir)
    valloader = load_val_data(valid_dir)
    
    # Continue tranining on old model?
    print(f"Train old model: {trained}")
    
    trained_model = None
    
    if(trained):
        load_model = load_checkpoint(save_dir)
        trained_model = load_model
    
    
    # Start training model
    training_model = train_model(output_features, trainloader, valloader, trained_model, trained, model_arch, lr, hidden_units, epochs, gpu)
    training_model.name = model_arch
    
    # Save model
    save_checkpoint(save_dir, training_model, train_data)
    
    load_model = load_checkpoint(save_dir)
    print(load_model)
    
    
    
    
    
    