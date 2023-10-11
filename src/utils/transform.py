import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix, matrix_to_quaternion


def xyz_axis_angle_to_matrix(xyz_axis_angle):
    assert isinstance(xyz_axis_angle, torch.Tensor)
    assert xyz_axis_angle.shape[-1] == 6

    mat = torch.zeros(xyz_axis_angle.shape[:-1] + (4, 4), dtype=xyz_axis_angle.dtype, device=xyz_axis_angle.device)
    mat[..., :3, :3] = axis_angle_to_matrix(xyz_axis_angle[..., 3:])
    mat[..., :3, 3] = xyz_axis_angle[..., :3]
    mat[..., 3, 3] = 1.
    assert mat.shape == xyz_axis_angle.shape[:-1] + (4, 4)
    # assert mat.shape[-2:] == (4, 4)
    return mat


def matrix_to_xyz_axis_angle(T):
    assert isinstance(T, torch.Tensor)
    assert T.dim() == 3
    assert T.shape[1:] == (4, 4)
    n_poses = len(T)
    q = matrix_to_quaternion(T[:, :3, :3])
    axis_angle = quaternion_to_axis_angle(q)
    xyz = T[:, :3, 3]
    poses = torch.concat([xyz, axis_angle], dim=1)
    assert poses.shape == (n_poses, 6)
    return poses
