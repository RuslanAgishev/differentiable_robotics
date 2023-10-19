import torch
import numpy as np
from utils.vis import draw_coord_frame, draw_bbox
from mayavi import mlab
from pytorch3d.transforms import euler_angles_to_matrix

def box_coverage(points, bbox_lhw, pose):
    # transform points to bbox frame
    pose_inv = torch.linalg.inv(pose)
    points_bbox_frame = points @ pose_inv[:3, :3].T + pose_inv[:3, 3:4].T

    # mask of points inside the cube
    mask = ((points_bbox_frame > -bbox_lhw / 2) & (points_bbox_frame < bbox_lhw / 2)).all(dim=1)

    # distance to the closest point on the cube surface
    dists = torch.min(torch.abs(points_bbox_frame - bbox_lhw / 2), torch.abs(points_bbox_frame + bbox_lhw / 2))
    dists = torch.sqrt(torch.square(dists).sum(dim=1))
    mask_soft = torch.exp(-torch.square(dists) / 2)

    return mask, mask_soft


def draw_cloud(points, colors, **kwargs):
    # points: (N, 3)
    # colors: (N, 3)
    assert points.shape[1] == 3
    assert colors.shape[1] == 3
    assert points.shape[0] == colors.shape[0]

    color_n = np.arange(len(points))
    lut = np.zeros((len(color_n), 4))
    lut[:, :3] = colors
    lut[:, 3] = 255

    p3d = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color_n, mode='point', **kwargs)
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
    p3d.module_manager.scalar_lut_manager.lut.table = lut

def demo():
    np.random.seed(0)
    torch.manual_seed(0)

    # bounding box parameters
    xyz_rpy_lhw = torch.tensor([1, 0.2, -1, 0., 0., np.pi/6, 1., 0.5, 0.2])
    xyz_rpy_lhw.requires_grad = True

    points = torch.as_tensor(np.random.randn(10000, 3), dtype=xyz_rpy_lhw.dtype)

    optimizer = torch.optim.Adam([xyz_rpy_lhw], lr=0.05)

    fig = mlab.figure(bgcolor=(0.5, 0.5, 0.5), size=(1000, 1000))
    draw_coord_frame(torch.eye(4), scale=1.)  # world frame
    # mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 1), scale_factor=0.1)
    # mlab.points3d(points[mask, 0], points[mask, 1], points[mask, 2], color=(0, 1, 0), scale_factor=0.1)
    for i in range(100):
        pose = torch.eye(4)
        R = euler_angles_to_matrix(xyz_rpy_lhw[3:6], 'XYZ')
        pose[:3, :3] = R
        pose[:3, 3] = xyz_rpy_lhw[:3]
        bbox_lhw = xyz_rpy_lhw[-3:]

        mask, mask_soft = box_coverage(points, bbox_lhw, pose)
        # print('Number of points inside the cube: ', mask.sum().item())
        # print('Total number of points: ', len(points))
        # print('Ratio: ', mask.float().mean().item())

        loss = 1. / mask_soft.mean()
        print('Loss: ', loss.item())

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mlab.clf()
            # visualize traj in mayavi
            draw_coord_frame(pose, scale=0.5)
            draw_bbox(bbox_lhw, pose)

            blue_tone = np.matmul((1 - mask_soft).numpy()[None].T, np.array([0, 0, 255])[None])
            green_tone = np.matmul(mask_soft.numpy()[None].T, np.array([0, 255, 0])[None])
            rgb = green_tone + blue_tone
            draw_cloud(points, rgb, opacity=1.)
            # set up view point
            mlab.view(azimuth=30, elevation=30, distance=15, focalpoint=[0, 0, 0])
            # save figure
            # mlab.savefig('box_coverage_%02d.png' % i)
            fig.scene._lift()


def main():
    demo()


if __name__ == '__main__':
    main()
