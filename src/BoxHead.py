import torch
import torch.cuda
import torch.nn.functional as F
from torch import nn
import torchvision.ops
from utils import *
from dataset import BuildDataset, BuildDataLoader
from pretrained_models import pretrained_models_680
from tqdm import tqdm
from torchvision import transforms
from main_visualization import *
import matplotlib

# matplotlib.use('TkAgg')

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



    # Post process the output for one image
    # Input:
    #       result_prob: (N,)
    #       result_class: (N,)
    #       box_decoded: (N, 4)     [x1,y1,x2,y2]
    #       IOU_thresh : scalar that is the IOU threshold for the NMS
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       post_nms_prob: (Post_NMS_boxes,)
    #       post_nms_class: (Post_NMS_boxes,)
    #       post_nms_box: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)

    def postprocess_detections(self, result_prob, result_class, box_decoded, IOU_thresh, conf_thresh, keep_num_preNMS,keep_num_postNMS):
        ######################################
        # TODO postprocess a image
        #####################################
        assert result_prob.dim() == 1
        assert result_class.dim() == 1
        assert box_decoded.dim() == 2
        assert box_decoded.shape[1] == 4

        nms_labels_list, nms_boxes_list, nms_prob_list = [],[],[]
        box_decoded[:, 0] = box_decoded[:, 0].clamp(min=0, max=1088 - 1)
        box_decoded[:, 1] = box_decoded[:, 1].clamp(min=0, max=800 - 1)
        box_decoded[:, 2] = box_decoded[:, 2].clamp(min=0, max=1088 - 1)
        box_decoded[:, 3] = box_decoded[:, 3].clamp(min=0, max=800 - 1)
        above_conf_prob, above_conf_class, above_conf_box= self.selectAboveConf(result_prob, result_class, box_decoded, conf_thresh)
        pre_nms_prob, pre_nms_class, pre_nms_box = selectResult(above_conf_prob, above_conf_class, above_conf_box,
                                                               N_keep=keep_num_preNMS)
        post_nms_labels_list = []
        post_nms_box = torch.zeros((0, 4), device=result_prob.device)
        post_nms_prob = torch.zeros(0, dtype=torch.float,  device=result_prob.device)
        for i in range(1,4):
            mask = (pre_nms_class == i)
            class_prob = pre_nms_prob[mask]
            class_box = pre_nms_box[mask,:]
            if class_prob.shape[0] == 0:
                continue
            nms_prob, nms_box = self.NMS(class_prob,class_box, IOU_thresh)
            assert nms_prob.dim() == 1
            assert nms_box.dim() == 2
            label = nms_prob.shape[0]*[i]

            post_nms_box = torch.cat((post_nms_box, nms_box), dim = 0)
            post_nms_prob = torch.cat((post_nms_prob, nms_prob))
            post_nms_labels_list += label

        post_nms_labels = torch.tensor(post_nms_labels_list)
        post_nms_prob, post_nms_class, post_nms_box = selectResult(post_nms_prob, post_nms_labels, post_nms_box,
                                                                N_keep=keep_num_postNMS)

        return post_nms_prob, post_nms_class, post_nms_box

    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform NSM
        ##################################
        assert clas.dim() == 1
        assert prebox.dim() == 2

        num_box = prebox.shape[0]
        if num_box == 1:
            return clas, prebox

        # IOU matrix
        iou_mat = torch.zeros((num_box, num_box),device=self.device)
        for x in range(num_box):
            for y in range(num_box):
                iou_mat[x, y] = IOU(torch.unsqueeze(prebox[x, :], 0), torch.unsqueeze(prebox[y, :], 0))
        max_index = set()

        # Suppressing small IOU
        for idx_curr in range(len(iou_mat)):    # w.r.t num_box
            # find all boxes with max iou from result list
            to_add = True
            to_remove = []                      # index to remove from result after matching
            for idx_prev in max_index:          # w.r.t. num_box
                # suppress bounding box when IOU > thres, only one lives
                if iou_mat[idx_curr, idx_prev] > thresh:
                    if clas[idx_curr] > clas[idx_prev]:             # add to remove list
                        to_remove.append(idx_prev)
                    else:
                        to_add = False                              # do not add if it's not the max one

            # all match done.
            for x in to_remove:                                     # do removal if any
                max_index.remove(x)
            if to_add:                                              # add if solo group / the global maxium
                max_index.add(idx_curr)

        if len(iou_mat) == len(max_index):      # quick return if keep everything
            return clas, prebox
        nms_clas = clas[list(max_index)]
        nms_prebox = prebox[list(max_index), :]

        assert nms_clas.dim() == 1
        assert nms_prebox.dim() == 2

        return nms_clas, nms_prebox

    def selectAboveConf(self,result_prob, result_class, box_decoded, conf_thresh):
        """
        :param prob_simp: (N,)
        :param class_simp: (N,)
        :param box_decoded: (N, 4)
        :return:
        """
        # remove background
        non_background_mask = result_class != 0
        class_obj = result_class[non_background_mask]
        prob_obj = result_prob[non_background_mask]
        box_obj = box_decoded[non_background_mask,:]

        # sort with descending order
        mask = (prob_obj > conf_thresh)
        above_conf_prob = prob_obj[mask]
        above_conf_class = class_obj[mask]
        above_conf_box = box_obj[mask,:]

        return above_conf_prob, above_conf_class, above_conf_box

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

if __name__ == '__main__':
    # reproductivity
    keep_topK_check = 20
    torch.random.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    batch_size = 1

    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"

    pretrained_path = '../pretrained/checkpoint680.pth'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths, augmentation=False)

    train_dataset, test_dataset = split_dataset(dataset)

    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader = train_build_loader.loader()

    # ============================ Train ================================
    box_head = BoxHead(device=device)
    box_head.to(device)

    result_dir = "../Grndbox"
    os.makedirs(result_dir, exist_ok=True)


    def color_switch(class_id):
        return{
            1: "b",
            2: "r",
            3: "g"
        }.get(class_id)


    for iter, data in enumerate(tqdm(train_loader), 0):
        img = data['images'].to(device)
        indexes = data['index']
        bbox_list = [x.to(device) for x in data['bbox']]
        label_list = [x.to(device) for x in data['labels']]
        num_bbox_class = []
        num_class1 = torch.count_nonzero(bbox_list[0] == 1)
        num_bbox_class.append(num_class1)
        num_class2 = torch.count_nonzero(bbox_list[0] == 2)
        num_bbox_class.append(num_class2)
        num_class3 = torch.count_nonzero(bbox_list[0] == 3)
        num_bbox_class.append(num_class3)

        image = transforms.functional.normalize(img[0].cpu().detach(),
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)

        image_vis = image.permute(1, 2, 0).cpu().detach().numpy()
        num_grnd_box = len(bbox_list)

        # Take the features from the backbone
        backout = backbone(img)

        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(img, [(800, 1088)] * img.shape[0])
        rpnout = rpn(im_lis, backout)

        # The final output is a list of proposal tensors: list:len(bz){(keep_topK,4)}
        proposals = [proposal[0:keep_topK_check, :] for proposal in rpnout[0]]
        # generate gt labels
        labels, regressor_target = box_head.create_ground_truth(proposals, label_list, bbox_list)      #tx,ty,tw,twh
        labels = labels.flatten()
        # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
        fpn_feat_list = list(backout.values())
        proposal_torch = torch.cat(proposals, dim=0)  # x1 y1 x2 y2
        proposal_xywh = torch.zeros_like(proposal_torch, device=proposal_torch.device)
        proposal_xywh[:, 0] = ((proposal_torch[:, 0] + proposal_torch[:, 2]) / 2)
        proposal_xywh[:, 1] = ((proposal_torch[:, 1] + proposal_torch[:, 3]) / 2)
        proposal_xywh[:, 2] = torch.abs(proposal_torch[:, 2] - proposal_torch[:, 0])
        proposal_xywh[:, 3] = torch.abs(proposal_torch[:, 3] - proposal_torch[:, 1])

        non_bg_mask = (labels != 0).flatten()
        label = labels[non_bg_mask]
        box_xywh = regressor_target[non_bg_mask,:]
        proposal_xy = proposal_torch[non_bg_mask,:]
        proposal_xywh = proposal_xywh[non_bg_mask,:]
        box_decoded = decode_output(proposal_xywh, box_xywh)

        for i in range(1,4):
            class_mask = (label == i)
            # if torch.count_nonzero(class_mask) == 0:
            #     continue
            class_grnd_bbox = box_decoded[class_mask, :]
            num_proposal_class = class_grnd_bbox.shape[0]
            class_proposal = proposal_xy[class_mask,:]
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image_vis)
            for num in range(num_proposal_class):
                coord = class_proposal[num,:]
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,                                             color='yellow')
                ax.add_patch(rect)

                coord = class_grnd_bbox[num,:]
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                         color=color_switch(i))
                ax.add_patch(rect)
            plt.savefig("{}/{}_class{}.png".format(result_dir,indexes[0],i))
            # plt.show()
            plt.close('all')

        if iter == 80:
            break







        # # plot bbox
        # rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
        #                          color='r')
        # ax.add_patch(rect)
        # # plot positive anchor
        # rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
        #                          fill=False, color='b')
        # ax.add_patch(rect)




