import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np

from renderer_pyrd_bkup import Renderer
from utils.geometry import batch_rodrigues
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

import torch
import torch.nn as nn


class pose_prediction(nn.Module):
    def __init__(self,
                 smpl,
                 frame_length=16,
                 input_pose_dim=72,
                 input_contact_dim=144,
                 output_pose_dim=144,
                 output_shape_dim=10,
                 output_transl_dim=3,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.2):
        super(pose_prediction, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_pose_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.shared_layers = nn.Sequential(*layers)

        self.pose_head = nn.Linear(hidden_dim, output_pose_dim)
        # self.shape_head = nn.Linear(hidden_dim, output_shape_dim)
        # self.transl_head = nn.Linear(hidden_dim, output_transl_dim)
        self.signature_head = nn.Linear(input_contact_dim, 75 * 75)
        self.segmentation_head = nn.Linear(input_contact_dim, 2 * 75)

        self.smpl = smpl

    def forward(self, x, gt):
        features = self.shared_layers(x)
        pred_pose6d = self.pose_head(features)
        # pred_shape = self.shape_output(features)
        # pred_transl = self.transl_head(features)

        pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1,6)).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)
        two_human_pose = torch.cat((x, pred_pose), dim=1)
        pred_segmentation = torch.sigmoid(self.segmentation_head(two_human_pose))
        pred_signature = torch.sigmoid(self.signature_head(two_human_pose))


        pred_vertices, pred_3dkp = self.smpl(
            gt["betas"].view(-1, 2, 10)[:, 1],
            pred_pose,
            gt["gt_cam_t"].view(-1, 2, )[:, 1],
            halpe=True
        )

        pred = {
            "pred_pose6d": pred_pose6d,
            "pred_pose": pred_pose,
            # "pred_shape": pred_shape,
            # "pred_transl": pred_transl,
            "pred_vertices": pred_vertices,
            "pred_3dkp": pred_3dkp,
            "pred_segmentation": pred_segmentation,
            "pred_signature": pred_signature,
        }

        return pred