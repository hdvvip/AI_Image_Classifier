from functions_col import import_func, process_image, imshow, predict, save_checkpoint, load_checkpoint, load_train_data, load_val_data, train_model, get_key_from_value
import argparse

import pandas as pd
import numpy as np

import torch
from torch import nn,optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session

from PIL import Image

import json

# Main 
if __name__ == '__main__':
    
    # Set up ArgumentParser to retrieve data from command line
    parser = argparse.ArgumentParser(
        description="AI Image Classfication App"
    )
    
    # Add parameters
    parser.add_argument('data_input', help="Path to Input File for ex: ..../image_07090.jpg")
    parser.add_argument('checkpoint', help="File Path where trained model saved to", default='checkpointTrainedModel.pth')
    parser.add_argument('--top_k', help="Top Predicted Classes", default=5, type=int)
    parser.add_argument('--category_names', help="file hold real name of classes", default='cat_to_name.json')
    parser.add_argument('--gpu', help="Train on GPU", action='store_true')
    
    
    #Parser the arguments
    args = parser.parse_args()
    
    # get values parsed from command line
    data_input = args.data_input
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    
    # Get real names
    cat_to_name = None
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # set up AUTOMATICALLY device to use is: gpu or cpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if(torch.cuda.is_available()):
            print("GPU available: YES \nRunning On: GPU \n")    
    else:
            print("GPU available: NO \nRunning On: CPU \n")  
    
#     if (gpu):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if(torch.cuda.is_available()):
#             ("GPU available: YES \n Training On: GPU \n")
#         else:
#             print("GPU available: NO \n Training On: CPU \n")
#     else:
#         device = torch.device("cpu")
    
    # Load model
    model = load_checkpoint(checkpoint)
    model.to(device)
    
    # Image path
    image_path = data_input
    # Get Image Label
    label = image_path.split('/')[2]

    # Predict the flower class based on iamge_path
    top_p, top_class = predict(image_path, model, device, top_k)
    
    # Map the predicted class to real label based on train_data.class_to_idx
    # The reason is: train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    # train_data map 0,1,2,3,4,5... to images based on the folder position from top to bottom
    # NOT the real label of image
    # Hence train_data.class_to_idx saved information about real label of the image
    # Therefore map the label 0,1,2,3,4,5... back to the real label of the iamge
    # model.class_to_idx == train_data.class_to_idx ==> TRUE
    lable_name_dictionary = model.class_to_idx
    top_class_mapped = [get_key_from_value(lable_name_dictionary, class_name) for class_name in top_class]
    
    # Get predicted top_class_name and real class name 
    top_class_name = [cat_to_name.get(class_name) for class_name in top_class_mapped]
#     label_name = cat_to_name.get(str(label))
    
    print(top_p)
    print(top_class_name)
    
    
    
