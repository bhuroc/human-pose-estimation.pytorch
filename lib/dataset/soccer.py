# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class SoccerDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(SoccerDataset, self).__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 23
        self.flip_pairs = [[0, 14], [1, 15], [2, 16], [3, 8], [4, 7], [5, 6], [10, 11], [9, 12],[17,21],[18,20]]
        self.parent_ids = None

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != '0test0':
                joints = np.array(a['joints'])
                #joints[:, 0:2] = joints[:, 0:2] - 1 <- matlab images are indexed from 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append({
                'image': os.path.join(self.root, image_dir, image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                })

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        return {'Null': 0.0}, 0.0
