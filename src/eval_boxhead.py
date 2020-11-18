import matplotlib
matplotlib.use('Agg')               # No display
import matplotlib.pyplot as plt
from matplotlib import patches
from pretrained_models import pretrained_models_680
from dataset import BuildDataset, BuildDataLoader
from utils import *
import os.path
import torch.backends.cudnn
import torch.utils.data
import numpy as np
from tqdm import tqdm

import utils
from BoxHead import BoxHead

def do_eval(dataloader, checkpoint_file, device, result_dir=None, keep_topK=200,keep_num_preNMS=50,keep_num_postNMS=5):

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs("PreNMS", exist_ok=True)
        os.makedirs("PostNMS", exist_ok=True)

    # =========================== Pretrained ===============================
    # Put the path were you save the given pretrained model
    pretrained_path = '../pretrained/checkpoint680.pth'
    backbone, rpn = pretrained_models_680(pretrained_path)
    backbone = backbone.to(device)
    rpn = rpn.to(device)
    # ========================= Loading Model ==============================
    boxHead = BoxHead(Classes=3, P=7, device=device).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    print("[INFO] Weight loaded from checkpoint file: {}".format(checkpoint_file))
    boxHead.load_state_dict(checkpoint['model_state_dict'])
    boxHead.eval()  # set to eval mode
    # ============================ Eval ================================
    for iter, data in enumerate(tqdm(dataloader), 0):
        img = data['images'].to(device)
        batch_size = img.shape[0]
        label_list = [x.to(device) for x in data['labels']]
        mask_list = [x.to(device) for x in data['masks']]
        bbox_list = [x.to(device) for x in data['bbox']]
        # index_list = data['index']
        img_shape = (img.shape[2], img.shape[3])
        with torch.no_grad():
            backout = backbone(img)
            im_lis = ImageList(img, [(800, 1088)] * img.shape[0])
            rpnout = rpn(im_lis, backout)
            proposals = [proposal[0:keep_topK, :] for proposal in rpnout[0]]
            fpn_feat_list = list(backout.values())
            feature_vectors = boxHead.MultiScaleRoiAlign(fpn_feat_list, proposals)
            class_logits, box_pred = boxHead(feature_vectors)
            class_logits = torch.softmax(class_logits, dim=1)
            proposal_torch = torch.cat(proposals, dim=0)  # x1 y1 x2 y2
            proposal_xywh = torch.zeros_like(proposal_torch, device=proposal_torch.device)
            proposal_xywh[:, 0] = ((proposal_torch[:, 0] + proposal_torch[:, 2]) / 2)
            proposal_xywh[:, 1] = ((proposal_torch[:, 1] + proposal_torch[:, 3]) / 2)
            proposal_xywh[:, 2] = torch.abs(proposal_torch[:, 2] - proposal_torch[:, 0])
            proposal_xywh[:, 3] = torch.abs(proposal_torch[:, 3] - proposal_torch[:, 1])
            result_prob, result_class, result_box = simplifyOutputs(class_logits, box_pred)
            box_decoded = decode_output(proposal_xywh, result_box)
            post_nms_prob, post_nms_class, post_nms_box= boxHead.postprocess_detections(result_prob, result_class,
                                            box_decoded,
                                           IOU_thresh=0.5,
                                           conf_thresh=0.5,
                                           keep_num_preNMS=keep_num_preNMS,keep_num_postNMS=keep_num_postNMS)

            # if iter == 4:
            #     break



if __name__ == '__main__':
    #reproductivity
    torch.random.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # =========================== Config ==========================
    batch_size = 1
    checkpoint_file = "checkpoints_sat/epoch_{}".format(49)
    assert os.path.isfile(checkpoint_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    keep_topK = 200
    USE_HOLD_OUT = True         # visualization of HOLD-OUT set

    # dir_prenms = "../results/preNMS"

    # =========================== Dataset ==============================
    # file path and make a list
    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    dataset = BuildDataset(paths, augmentation=False)
    train_dataset, test_dataset = utils.split_dataset(dataset)

    # dataset
    # train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    do_eval(test_loader, checkpoint_file, device, result_dir=None, keep_topK=200,keep_num_preNMS=50,keep_num_postNMS=5)