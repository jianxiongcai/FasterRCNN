"""
Run Pretrained RPN and savve all proposals
"""

import torch
from torchvision.models.detection.image_list import ImageList
import h5py
import numpy as np
from tqdm import tqdm

from pretrained_models import pretrained_models_680
from dataset import BuildDataset, BuildDataLoader
import utils

# ======================== Parameters =========================
imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"
pretrained_path = "../pretrained/checkpoint680.pth"

rpn_cache_file = "../data/rpn_cached.npy"
batch_size = 10
print("batch size:", batch_size)

# Here we keep the top 20, but during training you should keep around 200 boxes from the 1000 proposals
keep_topK = 200

# =========================== Code ============================\
def create_h5_dset(rpn_cache_file, N_images, N_proposal):
    """

    :param rpn_cache_file: The file to cache results
    :param N_images: number of images
    :param N_proposal: number of proposal per image
    :return:
        1. H5 File Object
        2. dictionary of h5 dataset objects
            - keys: 'proposal' | 'feat_0' | 'feat_1' | ... | 'feat_5'
            - values:
                + proposal: (N_images, N_proposal, 2)
                + feat_1: (N_images, 256, 200, 272)
                + feat_2: (N_images, 256, 100, 136)
    """
    h5_dict = {}
    feat_sizes_raw = [(256, 200, 272), (256, 100, 136), (256, 50, 68), (256, 25, 34), (256, 13, 17)]
    feat_sizes = []
    for i in range(len(feat_sizes_raw)):
        feat_sizes.append((N_images,) + feat_sizes_raw[i])

    h5_fd = h5py.File(rpn_cache_file, "w")
    h5_dict['proposal'] = h5_fd.create_dataset("proposal", (N_images, N_proposal, 4), chunks=True)
    for i in range(len(feat_sizes)):
        key = 'feat_{}'.format(i)
        h5_dict[key] = h5_fd.create_dataset(key, feat_sizes[i], chunks=True)
    return h5_fd, h5_dict


def save_rpn_result(proposals_np, fpn_feat_list_np, index, h5_dsets):
    """

    Note: Parameters are for each image
    :param proposals_np: list:len(bz){(keep_topK,4)}
    :param fpn_feat_list_np: list:len(FPN){(bz,256,H_feat,W_feat)}
    :param index: (bz,)
    :param h5_dsets:
    :return:
    """
    # flag check
    assert isinstance(proposals_np, list)
    assert isinstance(fpn_feat_list_np, list)
    assert isinstance(index, list)
    assert isinstance(proposals_np[0], np.ndarray)
    assert isinstance(fpn_feat_list_np[0], np.ndarray)

    bz = len(proposals_np)
    assert fpn_feat_list_np[0].shape[0] == bz
    assert len(index) == bz

    # save data
    for img_i in range(bz):
        idx = index[img_i]          # unique index of images[i]
        h5_dsets['proposal'][idx] = proposals_np[img_i]
        for fpn_level in range(5):
            h5_dsets['feat_{}'.format(fpn_level)][idx] = fpn_feat_list_np[fpn_level][img_i]


def infer_proposal(images, backbone, rpn, keep_topK):
    """

    :param images: image tensor from data loader
    :param backbone: the backbone network
    :param rpn: the rpn network
    :param keep_topK
    :return:
        - proposals: list:len(bz){(keep_topK,4)}
        - fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    """
    with torch.no_grad():
        # Take the features from the backbone
        backout = backbone(images)

        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(images, [(800, 1088)] * images.shape[0])
        # Then we pass the image list and the backbone output through the rpn
        rpnout = rpn(im_lis, backout)

        # The final output is
        # A list of proposal tensors: list:len(bz){(keep_topK,4)}
        proposals = [proposal[0:keep_topK, :] for proposal in rpnout[0]]
        # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
        fpn_feat_list = list(backout.values())

    return proposals, fpn_feat_list

# =========================== MAIN ============================
utils.set_deterministic()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
backbone, rpn = pretrained_models_680(pretrained_path)

# load the data into data.Dataset
paths = [imgs_path, masks_path, labels_path, bboxes_path]
dataset = BuildDataset(paths, augmentation=False)

# Run on the whole dataset
demo_build_loader = BuildDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
demo_loader = demo_build_loader.loader()

h5_fd, h5_dsets = create_h5_dset(rpn_cache_file, len(dataset), keep_topK)
for iter, batch in enumerate(tqdm(demo_loader), 0):
    images = batch['images'].to(device)
    index = batch['index']

    proposals, fpn_feat_list = infer_proposal(images, backbone, rpn, keep_topK)

    # convert to numpy
    proposals_np = list()                   # list:len(bz){(keep_topK,4)}
    fpn_feat_list_np = list()               # list:len(FPN){(bz,256,H_feat,W_feat)}
    for i in range(len(proposals)):
        proposals_np.append(proposals[i].cpu().detach().numpy())
    for i in range(5):
        fpn_feat_list_np.append(fpn_feat_list[i].cpu().detach().numpy())

    # save to h5
    save_rpn_result(proposals_np, fpn_feat_list_np, index, h5_dsets)

h5_fd.close()
print("[INFO] All DONE!")
