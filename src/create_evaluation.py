import torchvision
import torch
import numpy as np
from BoxHead import *
from utils import *
from pretrained_models import *

if __name__ == '__main__':

    # Put the path were you save the given pretrained model
    pretrained_path='../pretrained/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)
    backbone = backbone.to(device)
    rpn = rpn.to(device)

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    # Put the path were the given hold_out_images.npz file is save and load the images
    hold_images_path='../pretrained/hold_out_images.npz'
    test_images=np.load(hold_images_path,allow_pickle=True)['input_images']


    # Load your model here. If you use different parameters for the initialization you can change the following code
    # accordingly
    boxHead=BoxHead(device=device)
    boxHead=boxHead.to(device)
    boxHead.eval()

    # Put the path were you have your save network
    train_model_path='checkpoints_sat/epoch_49'
    print("[INFO] Loading from model: {}".format(train_model_path))
    checkpoint = torch.load(train_model_path)
    # reload models
    boxHead.load_state_dict(checkpoint['model_state_dict'])
    keep_topK=200

    cpu_boxes = []
    cpu_scores = []
    cpu_labels = []

    for i, numpy_image in enumerate(test_images, 0):
        images = torch.from_numpy(numpy_image).to(device)
        with torch.no_grad():
            # Take the features from the backbone
            backout = backbone(images)

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpnout = rpn(im_lis, backout)

            #The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list= list(backout.values())

            # do inference
            feature_vectors=boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)
            class_logits,box_pred=boxHead(feature_vectors)
            class_logits = torch.softmax(class_logits, dim=1)

            # convert proposal => xywh
            proposal_torch = torch.cat(proposals, dim=0)  # x1 y1 x2 y2
            proposal_xywh = torch.zeros_like(proposal_torch, device=proposal_torch.device)
            proposal_xywh[:, 0] = ((proposal_torch[:, 0] + proposal_torch[:, 2]) / 2)
            proposal_xywh[:, 1] = ((proposal_torch[:, 1] + proposal_torch[:, 3]) / 2)
            proposal_xywh[:, 2] = torch.abs(proposal_torch[:, 2] - proposal_torch[:, 0])
            proposal_xywh[:, 3] = torch.abs(proposal_torch[:, 3] - proposal_torch[:, 1])
            result_prob, result_class, result_box = simplifyOutputs(class_logits, box_pred)

            # decode box coordinate
            box_decoded = decode_output(proposal_xywh, result_box)

            # Do whaterver post processing you find performs best
            post_nms_prob, post_nms_class, post_nms_box=boxHead.postprocess_detections(result_prob, result_class, box_decoded, conf_thresh=0.8, keep_num_preNMS=200, keep_num_postNMS=3, IOU_thresh=0.5)

            post_nms_prob = [post_nms_prob]
            post_nms_class = [post_nms_class]
            post_nms_box = [post_nms_box]
            for box, score, label in zip(post_nms_box, post_nms_prob, post_nms_class):
                if box is None:
                    cpu_boxes.append(None)
                    cpu_scores.append(None)
                    cpu_labels.append(None)
                else:
                    cpu_boxes.append(box.to('cpu').detach().numpy())
                    cpu_scores.append(score.to('cpu').detach().numpy())
                    cpu_labels.append(label.to('cpu').detach().numpy())

    np.savez('predictions.npz', predictions={'boxes': cpu_boxes, 'scores': cpu_scores,'labels': cpu_labels})
