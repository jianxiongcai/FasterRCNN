import matplotlib
matplotlib.use('Agg')               # No display
import matplotlib.pyplot as plt
from matplotlib import patches

from dataset import BuildDataset, BuildDataLoader

import os.path
import torch.backends.cudnn
import torch.utils.data
import numpy as np
from tqdm import tqdm

import utils
from pretrained_models import pretrained_models_680
from BoxHead import *
from metric_tracker import MetricTracker
from utils import IOU

# ================================ Helpers ==========================================
def unnormalize_img(img):
    import torchvision.transforms.functional
    return torchvision.transforms.functional.normalize(img, mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                          std=[1/0.229, 1/0.224, 1/0.225])

def plot_prediction(img, class_selected, box_selected, index, result_dir):
    assert img.dim() == 4
    assert img.shape[0] == 1
    img_unnormalized = unnormalize_img(img)
    img_vis = img_unnormalized.detach().cpu().numpy()[0]
    img_vis = np.transpose(img_vis, (1, 2, 0))

    fig, ax = plt.subplots(1)
    ax.imshow(img_vis)

    # plot bounding box
    for i in range(class_selected.shape[0]):
        bbox = box_selected[i]
        if class_selected[i] == 1:
            color = 'b'
        elif class_selected[i] == 2:
            color = 'r'
        else:
            color = 'g'
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()
    plt.savefig(os.path.join(result_dir, "{}.png".format(index)))
    plt.close('all')


def selectResult(prob_simp, class_simp, box_decoded, N_keep = 20):
    """

    :param prob_simp: (N,)
    :param class_simp: (N,)
    :param box_decoded: (N, 4)
    :return:
    """

    # remove background
    class_obj = class_simp[class_simp != 0]
    prob_obj = prob_simp[class_simp != 0]
    box_obj = box_decoded[class_simp != 0]

    # sort with descending order
    prob_sorted, indices = torch.sort(prob_obj, descending=True)
    class_sorted = class_obj[indices]
    box_decoded_sorted = box_obj[indices]

    return prob_sorted[0:N_keep], class_sorted[0:N_keep], box_decoded_sorted[0:N_keep]


def run_inference(images, index, backbone, rpn, boxHead):
    """
    Run inference and visualization for one image
    :param images:
    :param index:
    :param backbone:
    :param rpn:
    :param boxHead:
    :return:
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

        feature_vectors = boxHead.MultiScaleRoiAlign(fpn_feat_list, proposals)

        class_logits, box_pred = boxHead(feature_vectors)
        class_logits = torch.softmax(class_logits, dim=1)  # todo: check softmax is applied everywhere

        # convert proposal to xywh
        proposal_torch = torch.cat(proposals, dim=0)  # x1 y1 x2 y2
        proposal_xywh = torch.zeros_like(proposal_torch, device=proposal_torch.device)
        proposal_xywh[:, 0] = ((proposal_torch[:, 0] + proposal_torch[:, 2]) / 2)
        proposal_xywh[:, 1] = ((proposal_torch[:, 1] + proposal_torch[:, 3]) / 2)
        proposal_xywh[:, 2] = torch.abs(proposal_torch[:, 2] - proposal_torch[:, 0])
        proposal_xywh[:, 3] = torch.abs(proposal_torch[:, 3] - proposal_torch[:, 1])

        # decode output
        prob_simp, class_simp, box_simp = utils.simplifyOutputs(class_logits, box_pred)
        # box_decoded: format x1, y1, x2, y2
        box_decoded = utils.decode_output(proposal_xywh, box_simp)

        # Do whaterver post processing you find performs best
        post_nms_prob, post_nms_class, post_nms_box = boxHead.postprocess_detections(prob_simp, class_simp, box_decoded, conf_thresh=0.8,
                                                               keep_num_preNMS=200, keep_num_postNMS=3, IOU_thresh=0.5)

        # visualization: PostNMS
        assert post_nms_class.dim() == 1
        assert post_nms_box.dim() == 2
        return post_nms_prob, post_nms_class, post_nms_box

# ===================================== MAIN ==================================================
def compute_map(dataloader, checkpoint_file, device):
    # =========================== Pretrained ===============================
    # Put the path were you save the given pretrained model
    pretrained_path = '../pretrained/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)
    backbone = backbone.to(device)
    rpn = rpn.to(device)

    # ========================== Prep ============================
    metric_trackers = []
    for i in range(3):
        metric_trackers.append(MetricTracker(i))

    # ========================= Loading Model ==============================
    boxHead = BoxHead(Classes=3, P=7, device=device).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    print("[INFO] Weight loaded from checkpoint file: {}".format(checkpoint_file))
    boxHead.load_state_dict(checkpoint['model_state_dict'])
    boxHead.eval()  # set to eval mode

    for iter, data in enumerate(tqdm(dataloader), 0):
        images = data['images'].to(device)
        assert len(images) == 1, "Only support batch_size == 1"

        labels_gt_all = data['labels'][0].to(device)
        bbox_gt_all = data["bbox"][0].to(device)
        index = data['index'][0]

        prob, clas, boxes = run_inference(images, index, backbone, rpn, boxHead)

        for tracker_i in range(3):
            # focus on one class
            target_class = tracker_i + 1

            labels_gt = labels_gt_all[labels_gt_all == target_class]
            bbox_gt = bbox_gt_all[labels_gt_all == target_class]  #n,4   x,y,w,h

            clas_pred = clas[clas == target_class]  #   m
            prob_pred = prob[clas == target_class]  #   m
            boxes_pred = boxes[clas == target_class]   #  m,4  x1,y1,x2,y2

            boxes_pred_xywh = torch.zeros_like(boxes_pred, dtype=boxes_pred.dtype, device = boxes_pred.device)
            boxes_pred_xywh[:,0]=(boxes_pred_xywh[:,0]+boxes_pred_xywh[:,2])/2
            boxes_pred_xywh[:,1]=(boxes_pred_xywh[:,1]+boxes_pred_xywh[:,3])/2
            boxes_pred_xywh[:,2]=boxes_pred_xywh[:,2]-boxes_pred_xywh[:,0]
            boxes_pred_xywh[:,3]=boxes_pred_xywh[:,3]-boxes_pred_xywh[:,1]

            # determine if it is a match with bbox
            N_gt = len(bbox_gt)  #n
            N_pred = len(clas_pred)  #m

            tp_indicator = torch.zeros((N_pred,))   #m
            match_indice = torch.zeros((N_pred,))   #m
            # IOU matrix
            iou_mat = torch.zeros((N_pred, N_gt))  # m,n
            for x in range(N_pred):
                for y in range(N_gt):
                    iou_mat[x, y] = IOU(torch.unsqueeze(boxes_pred_xywh[x, :], 0), torch.unsqueeze(bbox_gt[y, :], 0))
                if torch.max(iou_mat[x, :]) >= 0.5:
                    tp_indicator[x] = 1
                    index = torch.argmax(iou_mat[x, :])
                    match_indice[x] = index

            metric_trackers[tracker_i].add_match(prob_pred, tp_indicator, match_indice, N_gt)

        debug = True

    # compute map
    for i in range(3):
        metric_trackers[i].compute_precision_recall()
        recall, precision = metric_trackers[i].sorted_pr_curve()

        ap = metric_trackers[i].compute_ap()
        print("class_id: {}. ap: {}".format(i, ap))
        metric_trackers[i].reset()


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

    dir_prenms = "../results/preNMS"
    dir_postnms = "../results/postNMS"

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

    compute_map(test_loader, checkpoint_file, device)
