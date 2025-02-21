import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np

from renderer_pyrd_bkup import Renderer
from utils.geometry import batch_rodrigues
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

class pose_prediction(nn.Module):
    def __init__(self,
                 smpl,
                 frame_length=16,
                 input_pose_dim=72,
                 input_contact_dim=144,
                 output_pose_dim=144,
                 output_shape_dim=10,
                 output_transl_dim=3,
                 dropout=0.2):
        super(pose_prediction, self).__init__()

        self.shared_layers  = nn.Sequential(
            nn.Linear(input_pose_dim,  256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,  512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pose_head  = nn.Linear(512, output_pose_dim)
        # self.shape_head  = nn.Linear(512, output_shape_dim)
        # self.transl_head  = nn.Linear(512, output_transl_dim)

        self.segmentation_head = nn.Sequential(
            nn.Linear(input_contact_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * 34)
        )

        self.signature_head = nn.Sequential(
            nn.Linear(input_contact_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 34 * 34)
        )

        self.smpl  = smpl

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
        print(torch.max(pred_segmentation))
        print(torch.max(pred_signature))
        print(torch.min(pred_segmentation))
        print(torch.min(pred_signature))
        # 确保分割输出在 0~1 之间
        assert torch.all(pred_segmentation >= 0) and torch.all(pred_segmentation <= 1), "Segmentation output should be in [0,1] range"

        # 确保签名输出在 0~1 之间
        assert torch.all(pred_signature >= 0) and torch.all(pred_signature <= 1), "Signature output should be in [0,1] range"

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