import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np

from renderer_pyrd_bkup import Renderer
from utils.geometry import batch_rodrigues

import torch
import torch.nn as nn


class pose_prediction(nn.Module):
    def __init__(self, smpl, frame_length=16, input_dim=72, output_pose_dim=144, output_contact_dim=6890, hidden_dim=256, num_layers=3, dropout=0.2):
        super(pose_prediction, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.shared_layers = nn.Sequential(*layers)

        self.pose_output = nn.Linear(hidden_dim, output_pose_dim)
        self.contact_output = nn.Linear(hidden_dim, output_contact_dim)

        self.smpl = smpl

    def forward(self, x, gt):
        features = self.shared_layers(x)
        pred_pose = self.pose_output(features)
        pred_contact = self.contact_output(features)

        pred_pose = rot6d_to_rotmat(pred_pose).view(-1, 3, 3).detach().cpu().numpy()
        pose = []
        for i in range(len(pred_pose)):
            f_pose = cv2.Rodrigues(pred_pose[i])[0]
            pose.append(f_pose)
        pred_pose = torch.tensor(np.array(pose)).to("cuda").requires_grad_(True).view(-1, 72)
        pred_vertices, pred_3dkp = self.smpl(
            gt["betas"].view(-1, 2, 10)[:, 1],
            pred_pose,
            gt["gt_cam_t"].view(-1, 2, )[:, 1],
            halpe=True
        )

        pred_contact = torch.sigmoid(pred_contact)

        pred = {
            "pred_pose": pred_pose,
            "pred_vertices": pred_vertices,
            "pred_3dkp": pred_3dkp,
            "pred_contact": pred_contact
        }

        return pred

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
