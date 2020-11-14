import torch
import torch.cuda
import torch.nn.functional as F
from torch import nn
import torchvision.ops
from utils import *

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7,device=('cuda' if torch.cuda.is_available() else 'cpu')):
        super(BoxHead, self).__init__()
        self.C=Classes
        self.P=P
        # TODO initialize BoxHead
        self.device = device

        # define network
        self.interm_layer = nn.Sequential(
            nn.Linear(256 * P * P, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, self.C+1)
            # No softmax here. (Softmax performed in CrossEntropyLoss when computing loss)
            # nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(1024, self.C * 4),
        )

        # define loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

        self._init_weights()

    # This function initialize weights and bias for fully connected layer
    def _init_weights(self):
        for m in self.interm_layer.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                m.bias.data = torch.nn.init.constant_(m.bias.data, 0)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                m.bias.data = torch.nn.init.constant_(m.bias.data, 0)
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                m.bias.data = torch.nn.init.constant_(m.bias.data, 0)


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

        labels = torch.zeros(total_num_pro, dtype = torch.long).to(self.device)
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
        assert P == self.P, "[ERROR] Parameter does not agree with each other. P: {}, self.P: {}".format(P, self.P)
        bz = len(proposals)
        feat_vec_list = []

        device = proposals[0].device

        for img_i in range(bz):
            proposal_img = proposals[img_i]             # proposal within one image
            feat_vec = torch.zeros(len(proposal_img), 256 * P * P, device=device)

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
                prop_per_level = torch.zeros((N_prop_level, 5), device=device)
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
        assert isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor)

        # do sampling (class balancing ~3:1)
        class_logits_sampled, box_preds_sampled, \
        labels_sampled, regression_targets_sampled = self.do_sampling(class_logits, box_preds,
                                                                      labels, regression_targets,
                                                                      effective_batch)

        # cls loss
        loss_class = self.ce_loss(class_logits_sampled, torch.squeeze(labels_sampled, dim=1))

        # reg loss
        # take all non-background bbox
        box_list = [None, None, None]           # list: (num_pos_object, 4), only keep box correspond to gt object
        gt_box_list = [None, None, None]
        for gt_class in range(3):
            # 1-d binary indicator: if labels_sampled[i] == gt_class
            class_mask = (torch.squeeze(labels_sampled, dim=1) == (gt_class + 1))

            # selection range for class i: [i*4, i*4+4)
            box_selected = box_preds_sampled[class_mask, gt_class * 4: gt_class * 4 + 4]
            gt_box_selected = regression_targets[class_mask]

            box_list[gt_class] = box_selected
            gt_box_list[gt_class] = gt_box_selected

        # put all boxes together and compute loss
        box_regr = torch.cat(box_list, dim=0)
        gt_box_regr = torch.cat(gt_box_list, dim=0)
        loss_regr = self.smooth_l1_loss(box_regr, gt_box_regr)

        loss = loss_class * l * loss_regr
        return loss, loss_class, loss_regr

    def do_sampling(self, class_logits, box_preds, labels, regression_targets, effective_batch):
        """

        All params refer to compute_loss()
        :return:
        """
        # number of positive sample to keep after sampling
        N_pos = int(0.75 * effective_batch)

        # do positive sampling
        indicator_pos = torch.squeeze(labels != 0, dim=1)             # 1 dimensional
        indicator_neg = torch.squeeze(labels == 0, dim=1)
        indices_pos = torch.squeeze(torch.nonzero(indicator_pos), dim=1)
        indices_neg = torch.squeeze(torch.nonzero(indicator_neg), dim=1)

        if len(indices_pos) <= N_pos:       # take all available
            print("[WARN, minor] Not enough positive samples. Expected: {}, Actual: {}".format(N_pos, len(indices_pos)))
            N_pos = len(indices_pos)
            indices_keep_pos = indices_pos
        else:
            indices_keep_pos = self.random_choice(indices_pos, N_pos)

        # number of negative samples to keep
        N_neg = effective_batch - N_pos
        if len(indices_neg) < N_neg:       # take all available
            assert len(indices_neg) >= N_neg
            print("[WARN, major] Not enough negative samples. Expected: {}, Actual: {}".format(N_neg, len(indices_neg)))
            N_neg = len(indices_neg)
            indices_keep_neg = indices_neg
        else:
            indices_keep_neg = self.random_choice(indices_neg, N_neg)

        # return sampled batch
        return torch.cat([class_logits[indices_keep_pos], class_logits[indices_keep_neg]], dim=0), \
               torch.cat([box_preds[indices_keep_pos], box_preds[indices_keep_neg]], dim=0), \
               torch.cat([labels[indices_keep_pos], labels[indices_keep_neg]], dim=0), \
               torch.cat([regression_targets[indices_keep_pos], regression_targets[indices_keep_neg]], dim=0),

    @staticmethod
    def random_choice(x, N_sample):
        """

        :param x: the input tensor to select from
        :param N_sample: number of sample to keep
        :return:
        """
        idx = torch.randperm(len(x))
        return x[idx]

            # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):
        x = self.interm_layer(feature_vectors)

        class_logits = self.classifier(x)
        box_pred = self.regressor(x)

        assert class_logits.shape[1] == self.C + 1
        assert box_pred.shape[1] == self.C * 4

        return class_logits, box_pred

if __name__ == '__main__':
    pass