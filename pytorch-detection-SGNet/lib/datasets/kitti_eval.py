# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

MIN_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]

def parse_rec(filename):
    gts = []
    diffy = []
    with open(filename) as f:
        objs = [x.strip() for x in f.readlines()]

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            info = obj.split()

            obj_struct = {}
            obj_struct['name'] = info[0].strip()

            x1 = float(info[4])
            y1 = float(info[5])
            x2 = float(info[6])
            y2 = float(info[7])

            h = y2 - y1 + 1

            trunc = float(info[1])
            occ = int(info[2])

            difficult = 2
            if h >= MIN_HEIGHT[0] and occ == MAX_OCCLUSION[0] \
                and trunc >= 0 and trunc <= MAX_TRUNCATION[0]:
                difficult = 0
            elif h >= MIN_HEIGHT[1] \
                and (occ == MAX_OCCLUSION[0] or occ == MAX_OCCLUSION[1]) \
                and trunc >= 0 and trunc <= MAX_TRUNCATION[1]:
                difficult = 1

            obj_struct['difficult'] = difficult
            diffy.append(difficult)
            # Make pixel indexes 0-based
            obj_struct['bbox'] = [x1, y1, x2, y2]
            gts.append(obj_struct)

    return gts


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def kitti_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = kitti_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, 'annots.pkl')
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos_diffy = [0, 0, 0]
  for imagename in imagenames:

    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R])
    det = [False] * len(R)

    if difficult.shape[0] > 0:
      for level in np.nditer(difficult):
        npos_diffy[level] = npos_diffy[level] + 1

    #npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  l_tp = np.zeros((3, nd))
  l_fp = np.zeros((3, nd))

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['det'][jmax]:
          l_tp[ R['difficult'][jmax], d] = 1.
          R['det'][jmax] = 1
        else:
          l_fp[ R['difficult'][jmax], d] = 1.
      else:
        h = bb[3] - bb[1]
        if h >= MIN_HEIGHT[0]:
          l_fp[0, d] = 1.
        elif h >= MIN_HEIGHT[1]:
          l_fp[1, d] = 1.
        else:
          l_fp[2, d] = 1.

        

  l_rec = []
  l_prec = []
  l_ap = []

  for i in range(3):
    # compute precision recall
    fp = np.cumsum(l_fp[i])
    tp = np.cumsum(l_tp[i])
    rec = tp / float(npos_diffy[i])
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    l_rec.append(rec)
    l_prec.append(prec)
    l_ap.append(ap)

  return l_rec, l_prec, l_ap
