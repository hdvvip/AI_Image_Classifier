def import_func():
    # Imports here
    import pandas as pd
    import numpy as np

    import torch
    from torch import nn,optim

    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    from workspace_utils import active_session

    from PIL import Image

    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an TENSOR Array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    import torch
    from torch import nn,optim

    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    
    # WARNING!!!
    # The process_image function successfully converts a PIL image 
    # into an object that can be used as INPUT to a TRAINED model
    
    # A TRAINED model received Tensor Array not numpy array
    
    img = (image)
    
    transform_image = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
   
    
    transed_image =  transform_image(img)
    
    return transed_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    return ax


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    import torch
    from torchvision import datasets, transforms, models

    from PIL import Image
    
    # Open image from image_path
    image = Image.open(image_path)
    
    # Processed PIL Image to Tensor [3, 224, 224]
    image_processed = process_image(image)
    
    # image_path :[3, 244, 244] Convert to [1, 3, 224, 224]
    image_used = image_processed.view(1, 3, 224, 224)
    
    # Set up to run on gpu or cpu
    image_used = image_used.to(device)
    model = model.to(device)
    log_ps = 0
    ps = 0
    
    with torch.no_grad():
        # Feed forward image to model and get the probabilities
        log_ps = model.forward(image_used)

        # log_ps passed through Log Softmax, 
        # Hence, we do torch.exp() to get back probabilities
        ps = torch.exp(log_ps)
    
    
    
    # Get top proabiblities and class
    top_p, top_class = ps.topk(k=topk, dim=1)
    list_top_p, list_top_class = top_p.tolist()[0], top_class.tolist()[0]
      
    return list_top_p, list_top_class


def save_checkpoint(filepath , model, train_data):
    import torch
    
    checkpoint = {
    'class_to_idx': train_data.class_to_idx,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'model_name': model.name
    }

    torch.save(checkpoint, filepath)
    
    
def load_checkpoint(filepath):
    
    import torch
    from torch import nn,optim

    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    
    checkpoint = None
    
    if (torch.cuda.is_available()):
        # Checkpoint for when using GPU
        checkpoint = torch.load(filepath)
        
    else:
        # Checkpoint for when using CPU
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        
    
    model_name = checkpoint['model_name']
    
    model = models.__dict__[model_name](pretrained=True)
    
    
    #load info from check point to model
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model   


def load_train_data(train_dir, batch_size = 20):
    import torch
    from torch import nn,optim

    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    return trainloader, train_data
    
    
def load_val_data(val_dir, batch_size = 20):
    import torch
    from torch import nn,optim

    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    
    val_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    
    return valloader


def train_model(output_features, trainloader, valloader, trained_model, trained = False, model_arch = "vgg11", lr = 0.0008, hidden_units = 512, epochs = 1, gpu=True):
    import pandas as pd
    import numpy as np

    import torch
    from torch import nn,optim

    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    from workspace_utils import active_session
    
    # set up device to use is: gpu or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if(torch.cuda.is_available()):
            print("GPU available: YES \nTraining On: GPU \n")    
    else:
            print("GPU available: NO \nTraining On: CPU \n")    

    
#     if (gpu):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if(torch.cuda.is_available()):
#             ("GPU available: YES \n Training On: GPU \n")
#         else:
#             print("GPU available: NO \n Training On: CPU \n")
#     else:
#         device = torch.device("cpu")
    
    # Set up model
    model = None
    
    # Continue train on old model or starting training a new model
    if(trained):
        model = trained_model
    else:
        # get pretrain model, based on model_arch
        model = models.__dict__[model_arch](pretrained=True)

        # Freeze model parameter because we dont do backprop
        for params in model.parameters():
            params.requires_grad = False

        # Build new classifier for model
        # Get the input_features of the model to build the first hidden Layer
        input_features = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(hidden_units, output_features),
                                   nn.LogSoftmax(dim=1))

        # set own version classifier to model
        model.classifier = classifier

    
    # Loss function fomula: Negative Log Likely-Hood
    criterion = nn.NLLLoss()

    # run forward on classifier parameters ONLY
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        
    # Start training
    model = model.to(device)
    
    with active_session():
        # Start training neural network

        # Number of time to train neural network
        epochs = epochs
        training_loss = 0
        testing_loss = 0
        testing_accu = 0
        steps = 0
        print_every = 100

        for epoch in range(epochs):

            for features, labels in trainloader:
                
                # Move images, labels to gpu or cpu
                features, labels = features.to(device), labels.to(device)

                # Set gradient theta back to zero to avoid accumulation
                optimizer.zero_grad()

                log_ps = model.forward(features)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                # loss is tensor shape(1,)
                # loss.item() to get scalar inside loss
                training_loss += loss.item()
                steps += 1

                if(steps % print_every == 0):
                    # Turn on evaluation mode
                    model.eval()
                    with torch.no_grad():

                        for features, labels in valloader:
                            
                            features, labels = features.to(device), labels.to(device)

                            log_ps = model.forward(features)
                            loss = criterion(log_ps, labels)
                            testing_loss += loss.item()

                            # neural network model product LogSoftMax value
                            # Hence, torch.exp()  to get back the SoftMax probability
                            ps = torch.exp(log_ps)

                            # Get top probability and top_class
                            top_p, top_class = ps.topk(k=1, dim=1)

                            # Check difference between predicted class and real class
                            equals = labels == top_class.view(*labels.shape)

                            # equals is Byte Tensor, hence convert back to Float Tensor for mean calculation
                            # dont forget item() to get scalar value
                            testing_accu += torch.mean(equals.type(torch.FloatTensor)).item()


                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {training_loss/print_every:.3f}.. "
                          f"Test loss: {testing_loss/len(valloader):.3f}.. "
                          f"Test accuracy: {testing_accu/len(valloader):.3f}")


                    # Set everything back
                    steps = 0
                    training_loss = 0
                    testing_loss = 0
                    testing_accu = 0

                    # Turn on model training mode
                    model.train()
    
    return model
    
def get_key_from_value(dictionary, value_para):
    for key, value in dictionary.items():
        if(value_para == value):
            return key    