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
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

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
    def postprocess_detections(self, class_logits, box_regression, proposals, IOU_thresh=0.5, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        ######################################
        # TODO postprocess a batch of images
        #####################################
        bz = len(proposals)
        boxes = []
        scores = []
        labels = []
        for i in range(bz):
            proposal_each_img = proposals[i]
            cls_each_img = class_logits[i * len(proposal_each_img): (i+1)*len(proposal_each_img), :]
            box_each_img = box_regression[i * len(proposal_each_img): (i+1)*len(proposal_each_img), :]
            box, score, label = self.postprocessImg(cls_each_img, box_each_img, proposal_each_img, IOU_thresh, conf_thresh, keep_num_preNMS, keep_num_postNMS)
            boxes.append(box)
            scores.append(score)
            labels.append(label)

        return boxes, scores, labels

    # Post process the output for one image
    # Input:
    #       cls_each_img: (num_proposal_each_img,(C+1))
    #       box_each_img: (num_proposal_each_img,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposal_each_img: (num_per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       img_shape: tuple:len(2)
    #       conf_thresh: scalar
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,cls_each_img, box_each_img, proposal_each_img, IOU_thresh, conf_thresh, keep_num_preNMS, keep_num_postNMS):
            ######################################
            # TODO postprocess a single image
            #####################################
            for i in range(3):
                box_each_img[:, 4* i: 4*(i+1)] = output_decoding(box_each_img[:, 4* i : 4*(i+1)], proposal_each_img, device = self.device)

                # crop boundary crossing
                box_each_img[:, 4 * i] = box_each_img[:, 4 * i].clamp(min=0, max=1088 - 1)       #x1
                box_each_img[:, 4 * i + 1] = box_each_img[:, 4 * i + 1].clamp(min=0, max=800 - 1)  #y1
                box_each_img[:, 4 * i + 2] = box_each_img[:, 4 * i + 2].clamp(min=0, max=1088 - 1)  #x2
                box_each_img[:, 4 * i + 3] = box_each_img[:, 4 * i + 3].clamp(min=0, max=800 - 1)   #y2

            # prefilter
            sorted_scores, sorted_class = torch.sort(img_clas[:, 1:], descending=True)  # consider no-background
            sorted_scores, sorted_class = sorted_scores[:, 0], sorted_class[:, 0] + 1  # change class to {1,2,3}
            is_high_score = (sorted_scores > conf_thresh)  # suppress class confidence lower than threshold
            # only keep valid confidence score
            sorted_scores = sorted_scores[is_high_score]
            sorted_class = sorted_class[is_high_score]
            sorted_box = img_regr[is_high_score]


            # preNMS
            scores_order = torch.argsort(sorted_scores, descending=True)  # (len(per_image_proposals),), int
            scores_order = scores_order[0:keep_num_preNMS]  # (keep_num_preNMS,), int
            sorted_scores = sorted_scores[scores_order]  # (keep_num_preNMS,), float
            sorted_class = sorted_class[scores_order]  # (keep_num_preNMS,), int, {1,2,3}
            sorted_box = sorted_box[scores_order]  # (keep_num_preNMS, 4), float

            # NMS (for each class separately)
            score, box, label = [], [], []
            for cls in range(1, self.C + 1):
                cls_ind = (sorted_class == cls)
                label.append(torch.ones_like(sorted_scores[cls_ind]) * cls)
                score.append(sorted_scores[cls_ind])
                box.append(sorted_box[cls_ind, 4 * (cls - 1):4 * cls])

            each_class_cnt = [len(elem) for elem in score]
            while True:
                is_nms_finish = True
                for cls in range(1, self.C + 1):
                    nms_score, nms_box = self.NMS(score[cls - 1], box[cls - 1], IOU_thresh)
                    is_nms_finish *= (each_class_cnt[cls - 1] == len(nms_box))
                    each_class_cnt[cls - 1] = len(nms_box)
                    label[cls - 1] = torch.ones_like(nms_score) * cls
                    score[cls - 1] = nms_score
                    box[cls - 1] = nms_box
                if is_nms_finish: break
            score = torch.cat((score[0], score[1], score[2]))
            box = torch.cat((box[0], box[1], box[2]))
            label = torch.cat((label[0], label[1], label[2]))
            order = torch.argsort(score, descending=True)
            score = score[order][0:keep_num_postNMS]
            box = box[order][0:keep_num_postNMS]
            label = label[order][0:keep_num_postNMS]

            # suppress low score
            is_valid = (score > score_thresh)
            score = score[is_valid]
            box = box[is_valid]
            label = label[is_valid]

            return box, score, label













            reg_out = torch.unsqueeze(mat_coord, 0)
            cls_out = torch.unsqueeze(mat_clas, 0)
            flatten_coord, flatten_cls, flatten_anchors = output_flattening(reg_out, cls_out, self.anchors)
            decoded_preNMS_flatten_box = output_decoding(flatten_coord, flatten_anchors)
            a = [torch.rand(flatten_coord.shape[0]) > 0.5 for i in range(4)]
            x_low_outbound = (decoded_preNMS_flatten_box[:, 0] < 0)
            y_low_outbound = (decoded_preNMS_flatten_box[:, 1] < 0)
            x_up_outbound = (decoded_preNMS_flatten_box[:, 2] > 1088)
            y_up_outbound = (decoded_preNMS_flatten_box[:, 3] > 800)
            a[0] = x_low_outbound
            a[1] = y_low_outbound
            a[2] = x_up_outbound
            a[3] = y_up_outbound
            outbound_flatten_mask = (torch.sum(torch.stack(a), dim=0) > 0)
            flatten_cls[outbound_flatten_mask] = 0

            top_values, top_indices = torch.topk(flatten_cls, keep_num_preNMS)
            last_value = top_values[-1]
            topk_mask = flatten_cls >= last_value
            topk_cls = flatten_cls[topk_mask]
            topk_box = decoded_preNMS_flatten_box[topk_mask]
            pre_nms_dir = "PreNMS"
            self.plot_imgae_NMS(topk_box, image, pre_nms_dir,index, "Pre", keep_num_preNMS)

            nms_clas, nms_prebox = self.NMS(topk_cls,topk_box, IOU_thresh)

            num = min(nms_prebox.shape[0],keep_num_postNMS)
            top_values, top_indices = torch.topk(nms_clas, num)
            last_value = top_values[-1]
            topk_mask = nms_clas >= last_value
            topn_cls = nms_clas[topk_mask]
            topn_box = nms_prebox[topk_mask]
            post_nms_dir = "PostNMS"
            self.plot_imgae_NMS(topn_box, image, post_nms_dir,index, "Post", keep_num_postNMS)

            return topn_cls, topn_box


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
            gt_box_selected = regression_targets_sampled[class_mask]

            box_list[gt_class] = box_selected
            gt_box_list[gt_class] = gt_box_selected

        # put all boxes together and compute loss
        box_regr = torch.cat(box_list, dim=0)
        gt_box_regr = torch.cat(gt_box_list, dim=0)
        loss_regr = self.smooth_l1_loss(box_regr, gt_box_regr) / len(labels_sampled)

        loss = loss_class + l * loss_regr
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
        return x[idx[0:N_sample]]

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
