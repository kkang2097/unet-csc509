import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List
from skimage.transform import resize
from functools import partial
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

#From Datapath -> Getting our necessary data IDs
#test_size: size of test dataset
def get_all_ids(data_path, test_size = 0.2):
    #Copy-pasted from Notebook 3: CSC509
    train_dir = [f.path for f in os.scandir(data_path) if f.is_dir()]
    def list_to_ids(dir:str):
        """
        Will convert the dir paths to ids by parsing the paths.
        dir: string, image dir paths in BRATS
        """
        x = []
        for i in range(0,len(dir)):
            x.append(dir[i].split('/')[-1])
        return x
    ids = list_to_ids(train_dir)
    train_ids, test_ids = train_test_split(ids,test_size=0.2)
    train_ids, val_ids = train_test_split(train_ids,test_size=0.2)

    return train_ids, val_ids, test_ids



# Loads numpy array of image
def get_nifti(path):
    nifti_img = nib.load(path)
    nifti_arr = nifti_img.get_fdata()
    x,y,z = nifti_arr.shape

    #Range of image is 1,2,3,4
    sample_nifti = resize((nifti_arr[:, :, z // 2].T / z), (128, 128))
    max = np.max(sample_nifti)
    min = np.min(sample_nifti)

    return (sample_nifti - min)/(max - min)

#Filters the NIFTI image into [1,2,3,4]
def filter_nifti(nifti_array: np.ndarray, filter: List[float]):
    img = np.rint(nifti_array*4)
    array_filter = np.array(filter)
    img[np.isin(img, array_filter)] = 10.
    img[img != 10.] = 0.
    img[img == 10.] = 1.
    #print(type(img))
    return img

#Individual NIFTI tensor
def get_nifti_item(data_id: str, data_path, modes: List[str]):
    nifti_list = []
    for mode in modes:
        final_path = Path(data_path, f'{data_id}', f'{data_id}_{mode}.nii.gz')
        nifti = get_nifti(final_path)
        nifti_list.append(nifti)
    return np.stack(nifti_list)

#Grouping everything
def get_tensors(ids: str, data_path, modes: List[str]):
    nifti_list = []
    for item in tqdm(ids):
        #Shift the axis so we can conveniently implement the U-Net model
        nifti_list.append(np.moveaxis(get_nifti_item(item, data_path, modes), 0,-1))
    return np.stack(nifti_list)

#Filter the masked images only!
#Mask_index = which index of the "modes" in our NIFTI tensor is the filter?
def bulk_mask_filter(data_tensor, filter, mask_index = -1):
    #Fix one argument so that we can bulk filter!
    fixed_nifti_filter = partial(filter_nifti, filter=filter)
    filtered_data_tensor = data_tensor
    filtered_data_tensor[:, :, :, mask_index] = np.apply_along_axis(fixed_nifti_filter, 0, filtered_data_tensor[:, :, :, mask_index])
    return filtered_data_tensor