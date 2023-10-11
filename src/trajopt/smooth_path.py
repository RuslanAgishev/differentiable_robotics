#! /usr/bin/env python

import numpy as np
import torch
from utils.transform import xyz_axis_angle_to_matrix, matrix_to_xyz_axis_angle


def path_smoothness(xyza, dt=1.):
    assert isinstance(xyza, torch.Tensor)
    assert xyza.ndim == 2 or (xyza.ndim == 3 and xyza.shape[1:] == (4, 4))  # (N, 6) or (N, 4, 4)
    if xyza.ndim == 3 and xyza.shape[1:] == (4, 4):
        xyza = matrix_to_xyz_axis_angle(xyza)

    n_poses = len(xyza)
    assert n_poses >= 4, 'Path must contain at least 4 poses to compute acceleration and jerk'
    # xyza is a tensor of shape (N, 6)
    # where N is the number of poses
    # and each pose is a 6D vector (x, y, z, ax, ay, az)
    assert isinstance(xyza, torch.Tensor)
    assert xyza.shape[-1] == 6

    # estimate linear and angular velocities
    d_xyza = torch.diff(xyza, dim=0) / dt
    # l1 = torch.sum(torch.square(d_xyza[:, :3]))
    l1_rot = torch.sum(torch.square(d_xyza[:, 3:]))

    # estimate linear and angular accelerations
    dd_xyza = torch.diff(d_xyza, dim=0) / dt
    l2 = torch.sum(torch.square(dd_xyza[:, :3]))
    l2_rot = torch.sum(torch.square(dd_xyza[:, 3:]))

    # 3-rd order smoothness
    ddd_xyza = torch.diff(dd_xyza, dim=0) / dt
    l3 = torch.sum(torch.square(ddd_xyza[:, :3]))
    # l3_rot = torch.sum(torch.square(ddd_xyza[:, 3:]))

    # # estimate headings
    # headings = torch.atan2(xyza[1:, 1] - xyza[:-1, 1], xyza[1:, 0] - xyza[:-1, 0])
    # d_headings = torch.diff(headings, dim=0)

    # estimate cost
    cost = l2 + l3 + l1_rot + l2_rot

    return cost


def get_traj(N=10):
    # define trajectory as tensor of shape (N, 6)
    # where N is the number of poses
    # and each pose is a 6D vector (x, y, z, ax, ay, az)

    poses = torch.tensor([np.eye(4) for _ in range(N)])
    poses[:, 0, 3] = torch.linspace(0, 1, N)
    noise = 0.1
    poses[:, 1, 3] = torch.sin(torch.linspace(0, 1, N) * np.pi) + torch.randn(N) * noise
    poses[:, 2, 3] = torch.cos(torch.linspace(0, 1, N) * np.pi) + torch.randn(N) * noise
    xyza = matrix_to_xyz_axis_angle(poses)

    assert xyza.shape == (N, 6)
    return xyza


def demo():
    from mayavi import mlab
    from utils.vis import draw_coord_frames

    xyza = get_traj(N=10)
    xyza.requires_grad_(True)

    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    xyza_init = xyza.detach().clone()
    # optimize trajectory to have it smooth
    optimizer = torch.optim.Adam([xyza], lr=0.01)
    n_iters = 100

    for i in range(n_iters):
        optimizer.zero_grad()
        loss = path_smoothness(xyza)
        loss.backward()
        optimizer.step()
        print('iter {}, loss: {:.4f}'.format(i, loss.item()))

        with torch.no_grad():
            # visualize traj in mayavi
            mlab.clf()
            mlab.title('iter {}, loss: {:.4f}'.format(i, loss.item()), size=0.5, color=(0, 0, 0))
            mlab.points3d(xyza[:, 0], xyza[:, 1], xyza[:, 2], color=(0, 1, 0), scale_factor=0.1)
            mlab.plot3d(xyza[:, 0], xyza[:, 1], xyza[:, 2], color=(0, 1, 0), tube_radius=0.01)
            # draw_coord_frames(xyz_axis_angle_to_matrix(xyza), scale=0.5)

            # draw initial traj
            mlab.points3d(xyza_init[:, 0], xyza_init[:, 1], xyza_init[:, 2], color=(0, 0, 1), scale_factor=0.1)
            mlab.plot3d(xyza_init[:, 0], xyza_init[:, 1], xyza_init[:, 2], color=(0, 0, 1), tube_radius=0.01)
            fig.scene._lift()
    mlab.show()


if __name__ == '__main__':
    demo()
