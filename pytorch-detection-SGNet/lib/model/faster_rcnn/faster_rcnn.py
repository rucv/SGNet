import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, super_classes, sp_cls_rng, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)

        self.sp_classes = super_classes
        self.n_sp_classes = len(super_classes)
        self.sp_classes_rng = sp_cls_rng

        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat_fn = self.RCNN_base(im_data)
        base_feat_det = self.RCNN_br_coarse(base_feat_fn)
        base_feat_fn = self.RCNN_br_fine(base_feat_fn)


        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_det, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        pooled_feat = self.do_ROIs(base_feat_det, rois)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute object classification probability
        sp_cls_score = self.RCNN_cls_score(pooled_feat)
        _, sp_cls_idx = sp_cls_score.max(1)

        # compute bbox offset
        bbox_delta_pred = self.RCNN_bbox_pred(pooled_feat)

        conv_matrix = self.get_conv_matrix()
        rois_super_label = 0
        if self.training:
            rois_super_label = conv_matrix[rois_label]

        base_feat_fn = torch.cat((base_feat_fn, base_feat_det), 1)

        bbox_delta_pred, pred_boxes = self.get_final_boxes(bbox_delta_pred, rois, sp_cls_idx, rois_super_label, im_info)

        boxes2pool = sp_cls_score.new(batch_size, pred_boxes.size()[1], 5).zero_()
        for i in range(batch_size):
            boxes2pool[i,:,0] = i
        boxes2pool[:,:,1:] = pred_boxes[:, :, :]

        pooled_feat = self.do_ROIs(base_feat_fn, boxes2pool)
        cls_score = self.RCNN_cls_fine(pooled_feat.view(pooled_feat.size(0), -1))

        cls_prob = 0
        cls_prob_comb = 0
        RCNN_loss_sp_cls = 0
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:

            RCNN_loss_sp_cls = F.cross_entropy(sp_cls_score, rois_super_label)
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_delta_pred, rois_target, rois_inside_ws, rois_outside_ws)

            #pdb.set_trace()
        else:
            cls_prob = F.softmax(cls_score, 1)

            sp_cls_idx = sp_cls_idx.view(-1, 1).expand( cls_score.size() )
            conv_matrix = conv_matrix.view(1, -1).expand( cls_score.size() )

            cls_score[sp_cls_idx.ne(conv_matrix)] = cls_score.min()
            cls_prob_comb = F.softmax(cls_score, 1)


            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            cls_prob_comb = cls_prob_comb.view(batch_size, rois.size(1), -1)
            #bbox_delta_pred = bbox_delta_pred.view(batch_size, rois.size(1), -1)

        return rois, (cls_prob_comb, cls_prob), pred_boxes, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_sp_cls, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    #def _find
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


    def get_conv_matrix(self):
        conv_matrix = torch.LongTensor(self.n_classes).zero_().cuda()
        for sp_cls in range(self.n_sp_classes):
            conv_matrix[self.sp_classes_rng[sp_cls]:self.sp_classes_rng[sp_cls+1]] = sp_cls
        return conv_matrix


    def do_ROIs(self, base_feat, roi):
        roi = Variable(roi)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, roi.view(-1, 5))
            grid_xy = _affine_grid_gen(roi.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, roi.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, roi.view(-1,5))
        return pooled_feat

    def get_final_boxes(self, deltas, roi, cls_pred, cls_ann, info):

        if not self.class_agnostic:
            # select the corresponding columns according to roi labels/cls predictions
            bbox_pred_view = deltas.view(deltas.size(0), int(deltas.size(1) / 4), 4)
            if self.training:
                bbox_pred_select = torch.gather(bbox_pred_view, 1, cls_ann.view(cls_ann.size(0), 1, 1).expand(cls_ann.size(0), 1, 4))
            else:
                bbox_pred_select = torch.gather(bbox_pred_view, 1, cls_pred.view(cls_pred.size(0), 1, 1).expand(cls_pred.size(0), 1, 4))
            deltas = bbox_pred_select.squeeze(1)

        boxes = roi.data[:, :, 1:5]

        # get the bbox
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = deltas.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(roi.size(0), -1, 4)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, roi.size(0))
            pred_boxes = clip_boxes(pred_boxes, info, roi.size(0))
        else:
            # Simply repeat the boxes, once for each class
            _ = torch.from_numpy(np.tile(boxes, (1, 1)))
            pred_boxes = _.cuda()

        return deltas, pred_boxes
