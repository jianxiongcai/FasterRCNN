import torch
import torch.nn.functional as F
from torch import nn
from utils import *

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7,device=('cuda' if torch.cuda.is_available() else 'cpu'),):
        super(BoxHead, self).__init__()
        self.C=Classes
        self.P=P
        # TODO initialize BoxHead
        self.device = device



    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:pytorch
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)

    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals, gt_labels, bbox):
        bz = len(proposals)
        total_num_pro = 0
        for i in range(bz):
            total_num_pro += proposals[i].shape[0]

        labels = torch.zeros(total_num_pro).to(self.device)
        regressor_target = torch.zeros(total_num_pro, 4).to(self.device)

        count = 0
        for bz_index, each_bz_pro in enumerate(proposals):
            x = each_bz_pro
            y = torch.zeros_like(x, dtype=torch.float, device=self.device)
            y[:, 0] = (x[:, 0] + x[:, 2]) / 2
            y[:, 1] = (x[:, 1] + x[:, 3]) / 2
            y[:, 2] = (x[:, 2] - x[:, 0])
            y[:, 3] = (x[:, 3] - x[:, 1])
            box_bz = bbox[bz_index].view(-1, 4)
            gt_label_bz = gt_labels[bz_index]
            print(gt_label_bz.shape)
            num_obj = box_bz.shape[0]
            num_pro = y.shape[0]

            for i in range(num_pro):
                cur_pro = y[i].view(1, -1)
                cur_pro_n = cur_pro.repeat(num_obj, 1)
                iou = IOU(cur_pro_n, box_bz)
                max_iou = torch.max(iou)
                max_iou_idx = torch.argmax(iou)
                if max_iou <= 0.5:
                    continue
                if max_iou > 0.5:
                    labels[count + i] = gt_label_bz[max_iou_idx]
                    regressor_target[count + i, 0] = (box_bz[max_iou_idx, 0] - y[i, 0]) / (y[i, 2] + 1e-9)
                    regressor_target[count + i, 1] = (box_bz[max_iou_idx, 1] - y[i, 1]) / (y[i, 3] + 1e-9)
                    regressor_target[count + i, 2] = torch.log(box_bz[max_iou_idx, 2] / y[i, 2])
                    regressor_target[count + i, 3] = torch.log(box_bz[max_iou_idx, 3] / y[i, 3])
            count += num_pro
        labels=torch.unsqueeze(labels,1)
        return labels, regressor_target



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################

        return feature_vectors



    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):

        return boxes, scores, labels




    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):

        return loss, loss_class, loss_regr



    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):

        return class_logits, box_pred

if __name__ == '__main__':
