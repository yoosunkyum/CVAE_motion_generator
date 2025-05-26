import numpy as np
import torch
import torch.nn.functional as F
import os

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
    qx = torch.sign(m[:,2,1] - m[:,1,2]) * torch.sqrt(max(torch.zeros_like(m[:,0,0]), 1.0 + m[:,0,0] - m[:,1,1] - m[:,2,2])) / 2
    qy = torch.sign(m[:,0,2] - m[:,2,0]) * torch.sqrt(max(torch.zeros_like(m[:,0,0]), 1.0 - m[:,0,0] + m[:,1,1] - m[:,2,2])) / 2
    qz = torch.sign(m[:,1,0] - m[:,0,1]) * torch.sqrt(max(torch.zeros_like(m[:,0,0]), 1.0 - m[:,0,0] - m[:,1,1] + m[:,2,2])) / 2

    quat = torch.stack([qw, qx, qy, qz], dim=1)
    return F.normalize(quat, dim=1)

def quaternion_to_euler(q):
    """입력 quaternion [w, x, y, z] → 오일러 각 [yaw, pitch, roll] (radians)"""
    w, x, y, z = q

    # Yaw (Z축 회전)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Pitch (Y축 회전)
    sinp = 2 * (w * y - z * x)
    if torch.abs(sinp) >= 1:
        pitch = torch.sign(sinp) * (torch.pi / 2)  # Gimbal lock
    else:
        pitch = torch.asin(sinp)

    # Roll (X축 회전)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    return torch.stack([yaw, pitch, roll])

def euler_to_quaternion(yaw, pitch, roll):
    """오일러 각 [yaw, pitch, roll] → quaternion [w, x, y, z]"""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z])

def remove_yaw_torch(q):
    """주어진 quaternion [w, x, y, z]에서 yaw 제거한 quaternion 반환"""
    q = torch.tensor(q, dtype=torch.float32)
    yaw, pitch, roll = quaternion_to_euler(q)
    yaw_zero = torch.tensor(0.0)
    q_new = euler_to_quaternion(yaw_zero, pitch, roll)
    return q_new


def quaternion_inverse(q):
    # q: (N, 4) [w, x, y, z]
    w, x, y, z = q.unbind(dim=-1)
    norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
    return torch.stack([w, -x, -y, -z], dim=-1) / norm_sq

def quaternion_multiply(q1, q2):
    # q1, q2: (N, 4)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def compute_q4(q1, q2, q3):
    N = q3.shape[0]
    q1_expand = q1.unsqueeze(0).expand(N, -1)  # (N, 4)
    q2_expand = q2.unsqueeze(0).expand(N, -1)  # (N, 4)
    
    # 모든 입력: (N, 4)
    q1_inv = quaternion_inverse(q1_expand)
    q_rel = quaternion_multiply(q1_inv, q2_expand)
    q4 = quaternion_multiply(q3, q_rel)
    return q4

def integrate_quaternion(q0, omega, dt):
    """
    q0: (..., 4) -- initial quaternion [w, x, y, z]
    omega: (..., 3) -- angular velocity [vx, vy, vz]
    dt: float -- time step
    """
    # omega as quaternion: [0, vx, vy, vz]
    omega_q = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)

    dq = quaternion_multiply(q0, omega_q) * 0.5
    q1 = q0 + dq * dt

    # normalize to keep unit quaternion
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    return q1

