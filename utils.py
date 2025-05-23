import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import os
import random

def lr_scheduler(lr_max, lr_min, epoch, period):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch/period * np.pi))

def beta_scheduler(beta, epoch, period):
    if (epoch % period) < 0.5 * period:
        return 2 * (epoch % period)/period * beta
    else:
        return beta
    
def get_unique_folder_name(base_path):
    """
    같은 이름의 폴더가 있으면 뒤에 _1, _2, ...를 붙여서 고유한 폴더 경로 반환
    """
    candidate = base_path
    counter = 1

    while os.path.exists(candidate):
        candidate = f"{base_path}_{counter}"
        counter += 1

    return candidate

def lpf(x, x_1, alpha):
    return (1-alpha) * x + alpha * x_1

#6D rotation represenations for neural network learning (https://arxiv.org/pdf/1812.07035)

def quat2rot6d(q: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternion [w, x, y, z] to 6D rotation representation
    Output 6D vector is [col1; col2] flattened
    Input:
        q: Tensor of shape [N, 4] in [w, x, y, z] format
    Output:
        Tensor of shape [N, 6]
    """
    # Normalize quaternion
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)

    # Compute rotation matrix
    R = torch.stack([
        1 - 2*(y**2 + z**2),     2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),           1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y),           2*(y*z + w*x),       1 - 2*(x**2 + y**2)
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))  # shape: [N, 3, 3]

    # Extract first and second columns: shape [N, 3, 2]
    col1 = R[..., :, 0]  # [N, 3]
    col2 = R[..., :, 1]  # [N, 3]

    # Flatten to [N, 6] in [col1, col2] order
    rot6d = torch.cat([col1, col2], dim=-1)

    return rot6d

def rot6d2quat(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to quaternion [w, x, y, z]
    Input:  rot_6d [N, 6]
    Output: quat [N, 4]
    """
    r1 = rot_6d[:, 0:3]
    r2 = rot_6d[:, 3:6]

    x = F.normalize(r1, dim=-1)
    r2_proj = (r2 * x).sum(dim=-1, keepdim=True) * x
    y = F.normalize(r2 - r2_proj, dim=-1)
    z = torch.cross(x, y, dim=-1)

    R = torch.stack([x, y, z], dim=-1)  # [N, 3, 3]

    # Convert rotation matrix to quaternion
    m = R
    qw = torch.sqrt(1.0 + m[:,0,0] + m[:,1,1] + m[:,2,2]) / 2
    qx = torch.sign(m[:,2,1] - m[:,1,2]) * torch.sqrt(1.0 + m[:,0,0] - m[:,1,1] - m[:,2,2]) / 2
    qy = torch.sign(m[:,0,2] - m[:,2,0]) * torch.sqrt(1.0 - m[:,0,0] + m[:,1,1] - m[:,2,2]) / 2
    qz = torch.sign(m[:,1,0] - m[:,0,1]) * torch.sqrt(1.0 - m[:,0,0] - m[:,1,1] + m[:,2,2]) / 2

    quat = torch.stack([qw, qx, qy, qz], dim=1)
    return F.normalize(quat, dim=1)
