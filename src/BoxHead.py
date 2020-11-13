import torch
import torch.nn.functional as F
from torch import nn
import torchvision.ops
from utils import *

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        self.C=Classes
        self.P=P
        # TODO initialize BoxHead

        # define network




    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bbox):

        return labels,regressor_target



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
        bz = len(proposals)
        feat_vec_list = []
        for img_i in range(bz):
            proposal_img = proposals[img_i]             # proposal within one image
            feat_vec = torch.zeros(len(proposal_img), 256 * P * P)

            # compute the scale
            W = torch.abs(proposal_img[:, 2] - proposal_img[:, 0])
            H = torch.abs(proposal_img[:, 3] - proposal_img[:, 1])
            assert W.dim() == 1
            K = torch.floor(4 + torch.log2(torch.sqrt(W * H)/ 224 + 1e-8))
            K = torch.clamp(K, 2, 5)
            # to feature map level index
            # K denotes which feature level to pool feature from
            K = K - 2

            # do rescaling w.r.t feature
            # First feature map has stride of 4, second stride of 8
            # K \in [0, 3]
            # prop_rescaled: (per_image_proposals, 4)
            rescale_ratio = torch.pow(2, K) * 4
            prop_rescaled = proposal_img / torch.unsqueeze(rescale_ratio, dim=1)

            # pool feature from feature map
            for level in range(5):
                # only contain proposal for this level (rescaled), extend to 5 dimension (required trochvision api)
                N_prop_level = torch.sum(K == level).item()
                prop_per_level = torch.zeros((N_prop_level, 5))
                prop_per_level[:, 0] = img_i
                prop_per_level[:, 1:5] = prop_rescaled[K == level]
                feat_vec[K == level] = torchvision.ops.roi_align(fpn_feat_list[level],
                                                                 prop_per_level, (P, P)).view(-1, 256 * P * P)
            feat_vec_list.append(feat_vec)
        feature_vectors = torch.cat(feat_vec_list, dim=0)

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
        raise NotImplementedError("")

        # return boxes, scores, labels




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
