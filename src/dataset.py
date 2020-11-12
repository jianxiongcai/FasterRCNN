import torch
import torchvision.transforms.functional
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path, augmentation):
        #############################################
        # Initialize  Dataset
        #############################################
        if (augmentation):
            raise RuntimeError("[ERROR] Not supported yet! => proposal caching.")

        self.augmentation = augmentation
        # all files
        imgs_path, masks_path, labels_path, bboxes_path = path
        self.imgs_path = imgs_path
        self.masks_path = masks_path

        # get dataset length
        with h5py.File(self.imgs_path, 'r') as images_h5:
            self.N_images = images_h5['data'].shape[0]

        # load dataset
        # all images and masks will be lazy read
        self.labels_all = np.load(labels_path, allow_pickle=True)
        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)
        #        self.images_h5 = h5py.File(imgs_path, 'r')
        #        self.masks_h5 = h5py.File(masks_path, 'r')
        #        self.labels_all = np.load(labels_path, allow_pickle=True)
        #        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)

        # As the mask are saved sequentially, compute the mask start index for each images
        n_objects_img = [len(self.labels_all[i]) for i in range(len(self.labels_all))]  # Number of objects per list
        self.mask_offset = np.cumsum(n_objects_img)  # the start index for each images
        # Add a 0 to the head. offset[0] = 0
        self.mask_offset = np.concatenate([np.array([0]), self.mask_offset])


    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img: (3, 800, 1088)  (3,h,w)
        # label: (n_obj, )
        # transed_mask: (n_box, 800, 1088)
        # transed_bbox: (n_box, 4), unnormalized bounding box coordinates (resized)
        # index: if data augmentation is performed, the index will be a virtual unique identify (i.e. += len(dataset))
    def __getitem__(self, index):
        ################################
        # return transformed images,labels,masks,boxes,index
        ################################
        # images
        images_h5 = h5py.File(self.imgs_path, 'r')
        masks_h5 = h5py.File(self.masks_path, 'r')
        img_np = images_h5['data'][index] / 255.0  # (3, 300, 400)
        img = torch.tensor(img_np, dtype=torch.float)

        # annotation
        # label: start counting from 1
        label = torch.tensor(self.labels_all[index], dtype=torch.long)
        # collect all object mask for the image
        mask_offset_s = self.mask_offset[index]
        mask_list = []
        for i in range(len(label)):
            # get the mask of the ith object in the image
            mask_np = masks_h5['data'][mask_offset_s + i] * 1.0
            mask_tmp = torch.tensor(mask_np, dtype=torch.float)
            mask_list.append(mask_tmp)
        # (n_obj, 300, 400)
        mask = torch.stack(mask_list)

        # prepare data and do augumentation (if set)
        bbox_np = self.bboxes_all[index]
        bbox = torch.tensor(bbox_np, dtype=torch.float)
        transed_img, transed_mask, transed_bbox, index = self.pre_process_batch(img, mask, bbox, index)

        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        # close files
        images_h5.close()
        masks_h5.close()

        return transed_img, label, transed_mask, transed_bbox, index



    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox, index):
        #######################################
        # apply the correct transformation to the images,masks,boxes
        ######################################
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)
        # image
        img = self.torch_interpolate(img, 800, 1066)  # (3, 800, 1066)
        img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
        img = F.pad(img, [11, 11])

        # mask: (N_obj * 300 * 400)
        mask = self.torch_interpolate(mask, 800, 1066)  # (N_obj, 800, 1066)
        mask = F.pad(mask, [11, 11])  # (N_obj, 800, 1088)

        # transfer bounding box
        # 1) from (x1, y1, x2, y2) => (c_x, c_y, w, h)
        centralized = torch.zeros_like(bbox)
        centralized[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0
        centralized[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0
        centralized[:, 2] = bbox[:, 2] - bbox[:, 0]
        centralized[:, 3] = bbox[:, 3] - bbox[:, 1]

        # 2) to the resized image
        trans_bbox = torch.zeros_like(centralized)
        trans_bbox[:, 0] = centralized[:, 0] / 400.0 * 1066.0 + 11  # c_x
        trans_bbox[:, 1] = centralized[:, 1] / 300.0 * 800.0  # c_y
        trans_bbox[:, 2] = centralized[:, 2] / 400.0 * 1066.0  # w
        trans_bbox[:, 3] = centralized[:, 3] / 300.0 * 800.0  # h

        # do augmentation (if set)
        if self.augmentation and (np.random.rand(1).item() > 0.5):
            # perform horizontally flipping (data augmentation)
            assert img.dim() == 3
            assert mask.dim() == 3
            ret_img = torch.flip(img, dims=[2])
            ret_mask = torch.flip(mask, dims=[2])
            # bbox transform for augmentation (flip horizontal axis)
            ret_bbox = trans_bbox.clone()
            ret_bbox[:, 0] = 1088.0 - ret_bbox[:, 0]
            # for aug, create a unique index for unique identification
            index = index + self.N_images
        else:
            ret_img = img
            ret_mask = mask
            ret_bbox = trans_bbox

        assert ret_img.squeeze(0).shape == (3, 800, 1088)
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]
        assert ret_bbox.shape[0] == ret_mask.shape[0]

        return ret_img, ret_mask, ret_bbox, index

    def __len__(self):
        return self.N_images

    @staticmethod
    def torch_interpolate(x, H, W):
        """
        A quick wrapper function for torch interpolate
        :return:
        """
        assert isinstance(x, torch.Tensor)
        C = x.shape[0]
        # require input: mini-batch x channels x [optional depth] x [optional height] x width
        x_interm = torch.unsqueeze(x, 0)
        x_interm = torch.unsqueeze(x_interm, 0)

        tensor_out = F.interpolate(x_interm, (C, H, W))
        tensor_out = tensor_out.squeeze(0)
        tensor_out = tensor_out.squeeze(0)
        return tensor_out




class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        out_batch = {}
        tmp_image_list = []
        label_list = []
        mask_list = []
        bbox_list = []
        indice = []
        for transed_img, label, transed_mask, transed_bbox, index in batch:
            tmp_image_list.append(transed_img)
            label_list.append(label)
            mask_list.append(transed_mask)
            bbox_list.append(transed_bbox)
            indice.append(index)

        out_batch['images'] = torch.stack(tmp_image_list, dim=0)
        out_batch['labels'] = label_list
        out_batch['masks'] = mask_list
        out_batch['bbox'] = bbox_list
        out_batch['index'] = indice

        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
    pass