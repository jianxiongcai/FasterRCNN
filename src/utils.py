import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
from functools import partial
import os.path

def set_deterministic():
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(17)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_dataset(dataset):
    data_folder = "../data"
    assert os.path.isdir(data_folder)
    train_indices_file = os.path.join(data_folder, "train_indices.npy")
    test_indices_file = os.path.join(data_folder, "test_indices.npy")

    if not os.path.isfile("../data/train_indices.npy"):
        print("[WARN] No train/test split indices found. Generating and saving to ../data")

        # Standard Dataloaders Initialization
        full_size = len(dataset)
        train_size = int(full_size * 0.8)
        test_size = full_size - train_size

        old_state = torch.get_rng_state()
        torch.random.manual_seed(1)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        torch.set_rng_state(old_state)

        # save indices to disk
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices
        np.save(train_indices_file, train_indices)
        np.save(test_indices_file, test_indices)
    else:
        print("[INFO] Loading train/test indices from ../data")
        train_indices = np.load(train_indices_file).tolist()        # python.list
        test_indices = np.load(test_indices_file).tolist()          # python.list

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, test_dataset


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):

    return iou


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):
    
    return box