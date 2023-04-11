import torch
from torch.utils.data import Dataset
import numpy as np
import models.image_utils as img_utils
from abc import ABC, abstractmethod
from typing import List
from sklearn.model_selection import train_test_split

class NIFTI_interface(Dataset):

    def __init__(self, root: str, modes: List[str], filter: List[float]):
        #Root is the directory where our dataset is stored
        self.root = root
        self.filter = filter
        self.modes = modes


    #Essential functions
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    #We probably don't need this abstract method
    # @abstractmethod
    # def print_info(self):
    #     pass

    def init_dataset(self, which_split):
        train_ids, val_ids, test_ids = img_utils.get_all_ids(self.root)
        '''
        #Get tensors
        train_data = img_utils.get_tensors(train_ids, self.root, self.modes)
        val_data = img_utils.get_tensors(val_ids, self.root, self.modes)
        test_data = img_utils.get_tensors(test_ids, self.root, self.modes)
        #Get filtered masks
        self.train_data = img_utils.bulk_mask_filter(train_data, self.filter)
        self.val_data = img_utils.bulk_mask_filter(val_data, self.filter)
        self.test_data = img_utils.bulk_mask_filter(test_data, self.filter)
        '''
        #This automatically normalizes our data for us
        if(which_split == "train"):
            self.data = img_utils.get_tensors(train_ids, self.root, self.modes)
            self.data = img_utils.bulk_mask_filter(self.data, self.filter)
        elif(which_split == "val"):
            self.data = img_utils.get_tensors(val_ids, self.root, self.modes)
            self.data = img_utils.bulk_mask_filter(self.data, self.filter)
        elif(which_split == "test"):
            self.data = img_utils.get_tensors(test_ids, self.root, self.modes)
            self.data = img_utils.bulk_mask_filter(self.data, self.filter)
        
        #print(self.data.shape)
        self.data = np.moveaxis(self.data, -1, 1)
        #print(self.data.shape)
        self.data = torch.tensor(self.data)
        #Don't forget to shuffle!
        self.data = self.data[torch.randperm(self.data.shape[0]),:,:]
        return

class NIFTI_single_folder(NIFTI_interface):
    def __init__(self, root: str, modes: List[str], filter: List[float], which_split: str):
        #Root is the directory where our dataset is stored
        #which_split: train, val, or test split
        self.root = root
        self.filter = filter
        self.modes = modes

        self.data = None
        self.init_dataset(which_split)

    def __getitem__(self, idx: int):
        x = self.data[idx, :-1, :, :]
        y = self.data[idx,-1, :, :]
        # print(x.shape)
        # print(y.shape)


        #Make "x" 3-D, make "y" 2-D
        sample = {'x': x, 'y': y}
        return sample

    def __len__(self):
        return self.data.shape[0]



#class NIFTI_single_folder(NIFTI_interface):

if __name__ == "__main__":
    root = "../images/Module1_BraTS/MICCAI_BraTS2020_TrainingData"
    dataset = NIFTI_single_folder(root, ["t1", "t2", "seg"], [1., 2., 4.], "test")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample.keys())
        exit()

    print("hello world")