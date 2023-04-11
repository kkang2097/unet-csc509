import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from models.dataloader import NIFTI_single_folder
from models.unet import UNet
from tqdm import tqdm
import wandb

def train_unet(model, train_dataset):

    #Params

    n_epochs = 50
    batch_size = 4
    learning_rate = 0.00005
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    model_watch = False


    #Comment: Only use wandb when we are sure we trained the model

    #Coming up with the code for training U-Net
    for epoch in tqdm(range(n_epochs)):
        for i, data in enumerate(train_loader, 0):
            x = data['x'].double()
            y = data['y'].long()#.reshape([batch_size, 128, 128, 1])

            #Actually I think CPU will be faster than GPU in this case
            #since the matrices are very small
            x = x.cuda()
            y = y.cuda()


            optimizer.zero_grad()

            # print(type(x[0,0,0,0].item()))
            # exit()
            outputs = model(x)
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()

            if(model_watch == True):
                wandb.log({"loss": loss.item()})
        print("Loss at iteration ", epoch, ": ", loss.item())

    #Save the model!
    PATH = './unet_wt.pth'
    torch.save(model.state_dict(), PATH)

    return







if __name__ == "__main__":


    #Get data
    root_dir = "./images/Module1_BraTS/MICCAI_BraTS2020_TrainingData"
    filter = [1., 2., 4.]
    #We need the segmentation mask to be the last mode
    modes = ["t1", "t2", "seg"]
    train_loader = NIFTI_single_folder(root_dir, modes, filter, "train")

    #Get U-Net
    num_channels = len(modes) - 1
    num_classes = 2
    model = UNet(num_channels, num_classes, bilinear=False)
    model = model.double()
    model = model.cuda()

    #Use wandb

    #Train U-Net
    train_unet(model, train_loader)
    #store state_dict of weights
    pass