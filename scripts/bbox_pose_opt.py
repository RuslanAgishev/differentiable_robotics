import torch
import numpy as np
from utils.vis import draw_coord_frame, draw_bbox
from mayavi import mlab
from pytorch3d.transforms import euler_angles_to_matrix


def sigmoid(x, k=1.):
    # x: input
    # k: slope rate
    return 1. / (1. + torch.exp(-k * x))

def box_coverage(points, bbox_pose, bbox_lhw, sigmoid_slope=1., return_mask=False):
    # points: (N, 3)
    # bbox_lhw: (3,)
    # pose: (4, 4)
    assert points.shape[1] == 3
    assert bbox_lhw.shape == (3,)
    assert bbox_pose.shape == (4, 4)

    # transform points to bbox frame
    pose_inv = torch.linalg.inv(bbox_pose)
    points_bbox_frame = points @ pose_inv[:3, :3].T + pose_inv[:3, 3:4].T

    # https://openaccess.thecvf.com/content/WACV2023/papers/Deng_RSF_Optimizing_Rigid_Scene_Flow_From_3D_Point_Clouds_Without_WACV_2023_paper.pdf
    s1 = sigmoid(-points_bbox_frame - bbox_lhw / 2, k=sigmoid_slope)
    s2 = sigmoid(-points_bbox_frame + bbox_lhw / 2, k=sigmoid_slope)
    rewards = (s2 - s1).prod(dim=1)

    if return_mask:
        # mask of points inside the box
        mask = ((points_bbox_frame > -bbox_lhw / 2) & (points_bbox_frame < bbox_lhw / 2)).all(dim=1)
        return rewards, mask

    return rewards


def demo():
    np.random.seed(0)
    torch.manual_seed(0)

    points = torch.as_tensor(np.load('../data/car_points.npy'))

    # bounding box parameters
    xyz_rpy_lhw = torch.tensor([1, 0.2, -1, 0., 0., np.pi / 6, 1., 0.5, 0.2], dtype=points.dtype)
    xyz_rpy_lhw.requires_grad = True

    optimizer = torch.optim.Adam([xyz_rpy_lhw], lr=0.1)

    fig = mlab.figure(bgcolor=(0.5, 0.5, 0.5), size=(1000, 1000))
    draw_coord_frame(torch.eye(4), scale=1.)  # world frame
    for i in range(100):
        pose = torch.eye(4)
        R = euler_angles_to_matrix(xyz_rpy_lhw[3:6], 'XYZ')
        pose[:3, :3] = R
        pose[:3, 3] = xyz_rpy_lhw[:3]
        bbox_lhw = xyz_rpy_lhw[-3:]

        rewards, mask = box_coverage(points, pose, bbox_lhw, return_mask=True)
        loss = 1. / (rewards.mean() + 1e-6)
        print('Loss: ', loss.item())

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mlab.clf()
            # visualize in mayavi
            draw_coord_frame(pose, scale=0.5)
            draw_bbox(bbox_lhw, pose)
            mlab.points3d(points[mask, 0], points[mask, 1], points[mask, 2], color=(0, 1, 0), scale_factor=0.05, opacity=1.)
            mlab.points3d(points[~mask, 0], points[~mask, 1], points[~mask, 2], color=(0, 0, 1), scale_factor=0.03, opacity=0.5)
            # set up view point
            mlab.view(azimuth=30, elevation=30, distance=15, focalpoint=[0, 0, 0])
            fig.scene._lift()

    mlab.show()


def main():
    demo()


if __name__ == '__main__':
    main()
