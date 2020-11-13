"""
Training Main
"""
import utils
from dataset import BuildDataset, BuildDataLoader
from BoxHead import BoxHead

import os.path
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import numpy as np
import wandb
from pretrained_models import pretrained_models_680
from tqdm import tqdm

#reproductivity
torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# =========================== Config ==========================
batch_size = 2
init_lr = 7e-4
num_epochs = 50
milestones = [8, 13]
loss_ratio = 4
RESULT_DIR = "checkpoints"
keep_topK_train = 200
keep_topK_test = 200

# =========================== Logging ==========================
def log(mode, logging_cls_loss, logging_reg_loss, logging_tot_loss, LOGGING):
    print('Epoch:{} Sum. {} total loss: {:.4f}, loss cls: {}, loss reg: {}'.format(mode, epoch, logging_tot_loss,
                                                                                      logging_cls_loss, logging_reg_loss))
    if LOGGING == "wandb":
        wandb.log({"{}/cls_loss".format(mode): logging_cls_loss,
                   "{}/reg_loss".format(mode): logging_reg_loss,
                   "{}/tot_loss".format(mode): logging_tot_loss}, step=epoch)

# w and b login
# LOGGING = ""
LOGGING = "wandb"
if LOGGING == "wandb":
    assert os.system("wandb login $(cat wandb_secret)") == 0
    wandb.init(project="hw4")
    wandb.config.update({
        'batch_size': batch_size,
        'init_lr': init_lr,
        'num_epochs': num_epochs,
        'loss_ratio': loss_ratio
    })
# =========================== Dataset ==============================

# file path and make a list
imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"

# Put the path where you save the given pretrained model
pretrained_path = '../pretrained/checkpoint680.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
backbone, rpn = pretrained_models_680(pretrained_path)

# we will need the ImageList from torchvision
from torchvision.models.detection.image_list import ImageList

paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths, augmentation=True)

full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
train_loader = train_build_loader.loader()

test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = test_build_loader.loader()

# ============================ Train ================================
box_head = BoxHead(device=device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  box_head = nn.DataParallel(box_head)
box_head.to(device)

optimizer = optim.Adam(box_head.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

os.makedirs(RESULT_DIR, exist_ok=True)

# watch with wandb
if LOGGING == "wandb":
    wandb.watch(box_head)
for epoch in range(num_epochs):
    box_head.train()
    train_cls_loss = 0.0
    train_reg_loss = 0.0
    train_tot_loss = 0.0

    # ============================== EPOCH START ==================================
    for iter, data in enumerate(tqdm(train_loader), 0):
        img = data['images'].to(device)
        bbox_list = [x.to(device) for x in data['bbox']]
        label_list = [x.to(device) for x in data['labels']]
        optimizer.zero_grad()

        # Take the features from the backbone
        backout = backbone(img)

        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(img, [(800, 1088)] * img.shape[0])
        rpnout = rpn(im_lis, backout)

        # The final output is a list of proposal tensors: list:len(bz){(keep_topK,4)}
        proposals = [proposal[0:keep_topK_train, :] for proposal in rpnout[0]]
        # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
        fpn_feat_list = list(backout.values())

        # generate gt labels
        labels, regressor_target = box_head.create_ground_truth(proposals, label_list, bbox_list)
        # del data, bbox_list, index_list
        # generate feature_vectors
        feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals)

        #forward model
        class_logits, box_pred = box_head(feature_vectors)
        # compute loss and optimize
        # set l = 4, the raw regression loss is normalized for each bounding box coordinate
        loss, loss_c, loss_r = box_head.compute_loss(
            class_logits, box_pred, labels, regressor_target, l=loss_ratio, effective_batch=32)

        # epoch 0 is reference epoch
        if epoch != 0:
            loss.backward()
            optimizer.step()

        # logging
        train_cls_loss += loss_c.item()
        train_reg_loss += loss_r.item()
        train_tot_loss += loss.item()
        if np.isnan(train_tot_loss):
            raise RuntimeError("[ERROR] NaN encountered at iter: {}".format(iter))
    # ================================= EPOCH END ==================================
    # logging per epoch
    # save to files
    log("train", train_cls_loss / len(train_loader), train_reg_loss / len(train_loader),
        train_tot_loss / len(train_loader), LOGGING=LOGGING)

    # do validation
    box_head.eval()
    test_cls_loss = 0.0
    test_reg_loss = 0.0
    test_tot_loss = 0.0
    for iter, data in enumerate(tqdm(test_loader), 0):
        img = data['images'].to(device)
        bbox_list = [x.to(device) for x in data['bbox']]
        label_list = [x.to(device) for x in data['labels']]
        with torch.no_grad():
            # Take the features from the backbone
            backout = backbone(img)

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(img, [(800, 1088)] * img.shape[0])
            rpnout = rpn(im_lis, backout)

            # The final output is a list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals = [proposal[0:keep_topK_test, :] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list = list(backout.values())

            # generate gt labels
            labels, regressor_target = box_head.create_ground_truth(proposals, label_list, bbox_list)
            # del data, bbox_list, index_list
            # generate feature_vectors
            feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals)

            # forward model
            class_logits, box_pred = box_head(feature_vectors)
            # compute loss and optimize
            # set l = 4, the raw regression loss is normalized for each bounding box coordinate
            loss, loss_c, loss_r = box_head.compute_loss(
                class_logits, box_pred, labels, regressor_target, l=loss_ratio, effective_batch=32)

            # logging
            test_cls_loss += loss_c.item()
            test_reg_loss += loss_r.item()
            test_tot_loss += loss.item()
    # logging per epoch
    log("test", test_cls_loss/len(test_loader), test_reg_loss/len(test_loader),
        test_tot_loss / len(test_loader), LOGGING=LOGGING)

    # save checkpoint
    path = '{}/epoch_{}'.format(RESULT_DIR, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': box_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    if epoch != 0:
        scheduler.step()
