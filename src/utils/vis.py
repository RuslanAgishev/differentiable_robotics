from mayavi import mlab
import torch


__all__ = [
    'draw_coord_frame',
    'draw_coord_frames'
]

def draw_coord_frame(pose, scale=0.5):
    t, R = pose[:3, 3], pose[:3, :3]
    # draw coordinate frame
    x_axis = torch.tensor([1., 0., 0.], dtype=pose.dtype)
    y_axis = torch.tensor([0., 1., 0.], dtype=pose.dtype)
    z_axis = torch.tensor([0., 0., 1.], dtype=pose.dtype)
    x_axis = R @ x_axis
    y_axis = R @ y_axis
    z_axis = R @ z_axis
    mlab.quiver3d(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color=(1, 0, 0), scale_factor=scale)
    mlab.quiver3d(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color=(0, 1, 0), scale_factor=scale)
    mlab.quiver3d(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color=(0, 0, 1), scale_factor=scale)


def draw_coord_frames(poses, scale=0.1):
    assert poses.ndim == 3
    assert poses.shape[1:] == (4, 4)

    for pose in poses:
        draw_coord_frame(pose, scale=scale)
