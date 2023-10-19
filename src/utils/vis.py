from mayavi import mlab
import torch


__all__ = [
    'draw_coord_frame',
    'draw_coord_frames',
    'draw_bbox',
    'draw_bboxes',
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

def draw_bbox(lwh=(1, 1, 1), pose=torch.eye(4), color=(0, 0, 0)):
    # plot cube vertices as points and connect them with lines
    # lwh: length, width, height
    # pose: (4 x 4) pose of the cube
    l, w, h = lwh
    vertices = torch.tensor([[l / 2, w / 2, h / 2],
                             [l / 2, w / 2, -h / 2],
                             [l / 2, -w / 2, -h / 2],
                             [l / 2, -w / 2, h / 2],
                             [-l / 2, w / 2, h / 2],
                             [-l / 2, w / 2, -h / 2],
                             [-l / 2, -w / 2, -h / 2],
                             [-l / 2, -w / 2, h / 2]])
    vertices = pose[:3, :3] @ vertices.T + pose[:3, 3:4]
    vertices = vertices.T
    lines = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])

    mlab.points3d(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color, scale_factor=0.1)
    for line in lines:
        mlab.plot3d(vertices[line, 0], vertices[line, 1], vertices[line, 2], color=color, tube_radius=0.01)

def draw_bboxes(lwhs, poses, colors=None):
    # plot multiple cubes
    # lwhs: list of tuples (l, w, h)
    # poses: list of (4 x 4) poses of the cubes
    # colors: list of colors for each cube
    if colors is None:
        colors = [(0, 0, 0) for _ in range(len(lwhs))]
    for lwh, pose, color in zip(lwhs, poses, colors):
        draw_bbox(lwh, pose, color)
