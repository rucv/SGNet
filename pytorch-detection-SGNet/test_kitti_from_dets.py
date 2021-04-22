# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import pickle

# import logger
from common_utils.logging import Logger
from datasets.kitti_eval_numba import get_official_eval_result
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class kitti_lite():
    def __init__(self, image_set, devkit_path=None):
        self._image_set = image_set
        if image_set == 'test':
            self._image_folder = 'testing'
        else:
            self._image_folder = 'training'
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, self._image_folder)



        self._classes = ('__background__',  # always index 0
                         'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc', 'DontCare')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
      


        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._devkit_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'kitti')

    def _get_kitti_results_file_template(self):
        # results/kitti/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join('results', 'kitti', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def eval_from_results(self):
        print('load dets...')
        boxes_for_eval = self.get_dets_from_results(self._load_image_set_index())
        print('load gts...')
        gt_annos = self.get_label_annos(os.path.join(
                                            self._data_path,
                                            'label_2' ),
                                        self._load_image_set_index()
                                        )
        pdb.set_trace()

        print(get_official_eval_result(gt_annos, boxes_for_eval, [0, 1, 2, 3, 4, 5, 6]))
        #print(get_official_eval_result(gt_annos, boxes_for_eval, 0))

        

    def get_label_annos(self, label_folder, image_ids=None):
        annos = []
        for idx in image_ids:
            label_filename = os.path.join(label_folder, idx + '.txt')
            annos.append(self.get_label_anno(label_filename))
        return annos

    def get_label_anno(self, label_path):
        annotations = {}
        annotations.update({
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': []
        })
        with open(label_path, 'r') as f:
            lines = f.readlines()
        # if len(lines) == 0 or len(lines[0]) < 15:
        #     content = []
        # else:
        content = [line.strip().split(' ') for line in lines]
        annotations['name'] = np.array([x[0] for x in content])
        annotations['truncated'] = np.array([float(x[1]) for x in content])
        annotations['occluded'] = np.array([int(x[2]) for x in content])
        annotations['alpha'] = np.array([float(x[3]) for x in content])
        annotations['bbox'] = np.array(
            [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
        # dimensions will convert hwl format to standard lhw(camera) format.
        annotations['dimensions'] = np.array(
            [[float(info) for info in x[8:11]] for x in content]).reshape(
                -1, 3)[:, [2, 0, 1]]
        annotations['location'] = np.array(
            [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
        annotations['rotation_y'] = np.array(
            [float(x[14]) for x in content]).reshape(-1)
        return annotations

    def get_dets_from_results(self, img_idx):
        dets = [{
            'name': [],
            'bbox': [],
            'score': []
        }] * len(img_idx)

        img_to_idx = {im_idx: idx for idx, im_idx in enumerate(img_idx)}
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_kitti_results_file_template().format(cls)
            with open(filename, 'r') as f:
                lines = f.readlines()

                content = [line.strip().split(' ') for line in lines]

                for det in content:
                    if float(det[1]) < 0.05:
                        continue
                    dets[ img_to_idx[det[0]] ]['name'].append(cls)
                    dets[ img_to_idx[det[0]] ]['bbox'].append(
                        [float(info) for info in det[2:]]
                    )
                    dets[ img_to_idx[det[0]] ]['score'].append(float(det[1]))

        print('convert dets to numpy...')
        for im_ind in range(len(img_idx)):
            dets[im_ind]['name'] = np.array(dets[im_ind]['name'])
            dets[im_ind]['bbox'] = np.array(dets[im_ind]['bbox'])
            dets[im_ind]['score'] = np.array(dets[im_ind]['score'])

        return dets

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  # save log to file
  parser.add_argument('--save_folder', default='saved_log/',
                      help='Directory for saving checkpoint models')


  args = parser.parse_args()

  # create log save folder
  if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

  sys.stdout = Logger(os.path.join(args.save_folder, 'log0.txt'))
  return args

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  print("start imdb")
  misc_tic = time.time()

  imdb_kitti = kitti_lite('val')
  imdb_kitti.eval_from_results()

  misc_toc = time.time()
  nms_time = misc_toc - misc_tic

  sys.stdout.write('im_detect: {:.3f}s   \r'.format(nms_time))
  sys.stdout.flush()


