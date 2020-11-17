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
def IOU(bbox_1, bbox_2):
    x_1up, y_1up, x_1l, y_1l = bbox_1[:, 0] - 0.5 * bbox_1[:, 2], bbox_1[:, 1] - 0.5 * bbox_1[:, 3], bbox_1[:,0] + 0.5 * bbox_1[:,2], bbox_1[:,1] + 0.5 * bbox_1[:,3]
    x_2up, y_2up, x_2l, y_2l = bbox_2[:, 0] - 0.5 * bbox_2[:, 2], bbox_2[:, 1] - 0.5 * bbox_2[:, 3], bbox_2[:,0] + 0.5 * bbox_2[:,2], bbox_2[:,1] + 0.5 * bbox_2[:,3]

    x_up = torch.max(x_1up, x_2up)
    y_up = torch.max(y_1up, y_2up)

    x_l = torch.min(x_1l, x_2l)
    y_l = torch.min(y_1l, y_2l)

    inter_area = (x_l - x_up).clamp(min=0) * (y_l - y_up).clamp(min=0)

    area_box1 = (x_1l - x_1up).clamp(min=0) * (y_1l - y_1up).clamp(min=0)
    area_box2 = (x_2l - x_2up).clamp(min=0) * (y_2l - y_2up).clamp(min=0)
    union_area = area_box1 + area_box2 - inter_area
    iou = (inter_area + 1e-9) / (union_area + 1e-9)

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


# =========================== Jianxiong Added ================================
def decode_output(proposal_xywh, box_xywh):
    """

    :param proposal_xywh: (N, 4): proposal in xywh format
    :param box_xywh:      (N, 4): boxes in xywh format
    :return:
        box_decoded: the decoded boxes (in format x1, y1, x2, y2)
    """
    assert proposal_xywh.shape[1] == 4
    assert box_xywh.shape[1] == 4

    box_decoded = torch.zeros_like(box_xywh, device=box_xywh.device)
    box_decoded[:, 0] = box_xywh[:, 0] * proposal_xywh[:, 2] + proposal_xywh[:, 0]
    box_decoded[:, 1] = box_xywh[:, 1] * proposal_xywh[:, 3] + proposal_xywh[:, 1]
    box_decoded[:, 2] = torch.exp(box_xywh[:, 2]) * proposal_xywh[:, 2]
    box_decoded[:, 3] = torch.exp(box_xywh[:, 3]) * proposal_xywh[:, 3]

    return box_decoded


def simplifyOutputs(class_logits, box_pred):
    """
    For each prediction, take the maximal prob and its corresponding box
    All background is going to have
    :param class_logits: N * (C+1)
    :param box_pred: N * (4 * C), [tx ty th tw], [tx ty th tw], [tx ty th tw]
    :return:
        result_prob: (N,)
        result_class: (N,)
        result_box: (N, 4), tx ty th tw
    """
    assert class_logits.shape[1] == 4
    assert box_pred.shape[1] == 12

    N = class_logits.shape[0]
    result_prob = torch.zeros((N,), device=class_logits.device)
    result_class = torch.zeros((N,), device=class_logits.device)
    result_box = torch.zeros((N,4), device=class_logits.device)

    # compute the indice and max prob
    tmp = torch.max(class_logits, dim=1)
    result_prob = tmp.values
    result_class = tmp.indices

    # get the corresponding box
    for i in range(3):
        result_box[result_class == (i+1)] = box_pred[result_class == (i+1), (i*4): (i*4 + 4)]

    return result_prob, result_class, result_box
