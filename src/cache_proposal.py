import torch

from pretrained_models import pretrained_models_680
from dataset import BuildDataset, BuildDataLoader
import utils

# ======================== Parameters =========================
imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"
pretrained_path = "../pretrained/checkpoint680.pth"


# =========================== Code ============================
utils.set_deterministic()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
backbone, rpn = pretrained_models_680(pretrained_path)

# load the data into data.Dataset
paths = [imgs_path, masks_path, labels_path, bboxes_path]
dataset = BuildDataset(paths, augmentation=False)
