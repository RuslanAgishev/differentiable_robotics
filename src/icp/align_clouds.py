from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from pytorch3d.ops.knn import knn_points
import warnings


__all__ = [
    'point_to_point_dist',
    'point_to_plane_dist'
]


def point_to_plane_dist(clouds: list, icp_inlier_ratio=0.9, masks=None, differentiable=True, verbose=False, **kwargs):
    """ICP-like point to plane distance.

    Computes point to plane distances for consecutive pairs of point cloud scans, and returns the average value.

    :param clouds: List of clouds. Individual scans from a data sequences.
    :param masks: List of tuples masks[i] = (mask1, mask2) where mask1 defines indices of points from 1st point cloud
                  in a pair that intersect (close enough) with points from 2nd cloud in the pair,
                  mask2 is list of indices of intersection points from the 2nd point cloud in a pair.
    :param icp_inlier_ratio: Ratio of inlier points between a two pairs of neighboring clouds.
    :param differentiable: Whether to use differentiable method of finding neighboring points (from Pytorch3d: slow on CPU)
                           or from scipy (faster but not differentiable).
    :param verbose:
    :return:
    """
    assert 0.0 <= icp_inlier_ratio <= 1.0
    if masks is not None:
        assert len(clouds) == len(masks) + 1
        # print('Using precomputed intersection masks for point to plane loss')
    point2plane_dist = 0.0
    n_pairs = len(clouds) - 1
    for i in range(n_pairs):
        cloud1 = clouds[i]
        assert cloud1.normals is not None, "Cloud must have normals computed to estimate point to plane distance"
        cloud2 = clouds[i + 1]

        points1 = cloud1.to_points() if cloud1.points is None else cloud1.points
        points2 = cloud2.to_points() if cloud2.points is None else cloud2.points
        assert not torch.all(torch.isnan(points1))
        assert not torch.all(torch.isnan(points2))
        points1 = torch.as_tensor(points1, dtype=torch.float)
        points2 = torch.as_tensor(points2, dtype=torch.float)

        # find intersections between neighboring point clouds (1 and 2)
        if masks is None:
            if not differentiable:
                tree = cKDTree(points2)
                dists, ids = tree.query(points1, k=1)
            else:
                dists, ids, _ = knn_points(points1[None], points2[None], K=1)
                dists = torch.sqrt(dists).squeeze()
                ids = ids.squeeze()
            dists = torch.as_tensor(dists)
            dist_th = torch.nanquantile(dists, icp_inlier_ratio)
            mask1 = dists <= dist_th
            mask2 = ids[mask1]
            inl_err = dists[mask1].mean()
        else:
            mask1, mask2 = masks[i]
            inl_err = torch.tensor(-1.0)

        points1_inters = points1[mask1]
        assert len(points1_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"
        points2_inters = points2[mask2]

        # point to plane distance 1 -> 2
        normals1_inters = cloud1.normals[mask1]
        # assert np.allclose(np.linalg.norm(normals1_inters, axis=1), np.ones(len(normals1_inters)))
        k = torch.multiply(normals1_inters, points2_inters - points1_inters).sum(dim=-1, keepdims=True)
        points2_plane = points2_inters - k * normals1_inters
        dists_to_plane = torch.linalg.norm(points2_inters - points2_plane, dim=-1)
        dist12 = dists_to_plane.mean()

        # point to plane distance 2 -> 1
        normals2_inters = cloud2.normals[mask2]
        # assert np.allclose(np.linalg.norm(normals2_inters, axis=1), np.ones(len(normals2_inters)))
        k = torch.multiply(normals2_inters, points1_inters - points2_inters).sum(dim=-1, keepdims=True)
        points1_plane = points1_inters - k * normals2_inters
        dists_to_plane = torch.linalg.norm(points1_inters - points1_plane, dim=-1)
        dist21 = dists_to_plane.mean()

        point2plane_dist += 0.5 * (dist12 + dist21)

        if inl_err > 0.3:
            warnings.warn('ICP inliers error is too big: %.3f (> 0.3) [m] for pairs (%i, %i)' % (inl_err, i, i + 1))

        if verbose:
            print('Mean point to plane distance: %.3f [m] for scans: (%i, %i), inliers error: %.6f' %
                  (point2plane_dist.item(), i, i+1, inl_err.item()))

    point2plane_dist = torch.as_tensor(point2plane_dist / n_pairs)

    return point2plane_dist


def point_to_point_dist(clouds: list, icp_inlier_ratio=0.9, differentiable=True, verbose=False):
    """ICP-like point to point distance.

    Computes point to point distances for consecutive pairs of point cloud scans, and returns the average value.

    :param clouds: List of clouds. Individual scans from a data sequences.
    :param icp_inlier_ratio: Ratio of inlier points between a two pairs of neighboring clouds.
    :param verbose:
    :return:
    """
    assert 0.0 <= icp_inlier_ratio <= 1.0

    point2point_dist = 0.0
    n_pairs = len(clouds) - 1
    for i in range(n_pairs):
        points1 = clouds[i]
        points2 = clouds[i + 1]

        points1 = torch.as_tensor(points1, dtype=torch.float)
        points2 = torch.as_tensor(points2, dtype=torch.float)
        assert not torch.all(torch.isnan(points1))
        assert not torch.all(torch.isnan(points2))

        # find intersections between neighboring point clouds (1 and 2)
        if not differentiable:
            tree = cKDTree(points2)
            dists, ids = tree.query(points1.detach(), k=1)
        else:
            dists, ids, _ = knn_points(points1[None], points2[None], K=1)
            dists = torch.sqrt(dists).squeeze()
            ids = ids.squeeze()
        dists = torch.as_tensor(dists)
        dist_th = torch.nanquantile(dists, icp_inlier_ratio)
        mask1 = dists <= dist_th
        mask2 = ids[mask1]
        inl_err = dists[mask1].mean()

        points1_inters = points1[mask1]
        assert len(points1_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"
        points2_inters = points2[mask2]
        assert len(points2_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"

        # point to point distance
        vectors = points2_inters - points1_inters
        point2point_dist = torch.linalg.norm(vectors, dim=1).mean()

        if inl_err > 0.3:
            print('ICP inliers error is too big: %.3f (> 0.3) [m] for pairs (%i, %i)' % (inl_err, i, i + 1))

        if verbose:
            print('Mean point to point distance: %.3f [m] for scans: (%i, %i), inliers error: %.6f' %
                  (point2point_dist.item(), i, i+1, inl_err.item()))

    # point2point_dist = torch.as_tensor(point2point_dist / n_pairs)
    return point2point_dist


def clouds_alignment_demo():
    from utils.preproc import filter_grid, filter_depth
    from utils.transform import matrix_to_xyz_axis_angle, xyz_axis_angle_to_matrix, transform_cloud
    from utils.io import read_cloud, read_poses
    import open3d as o3d
    from matplotlib import pyplot as plt
    from numpy.lib.recfunctions import structured_to_unstructured

    id1, id2 = '1669300804_715071232', '1669300806_15306496'
    # load cloud poses
    poses = read_poses(os.path.join('poses.csv'))
    pose1 = poses[id1]
    pose2 = poses[id2]

    # load point clouds
    cloud1 = read_cloud('%s.npz' % id1)
    cloud2 = read_cloud('%s.npz' % id2)
    cloud1 = structured_to_unstructured(cloud1[['x', 'y', 'z']])
    cloud2 = structured_to_unstructured(cloud2[['x', 'y', 'z']])

    # apply grid filtering to point clouds
    cloud1 = filter_grid(cloud1, grid_res=0.1)
    cloud2 = filter_grid(cloud2, grid_res=0.1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cloud1 = torch.as_tensor(cloud1, dtype=torch.float32, device=device)
    cloud2 = torch.as_tensor(cloud2, dtype=torch.float32, device=device)
    pose1 = torch.tensor(pose1, dtype=torch.float32, device=device)
    pose2 = torch.tensor(pose2, dtype=torch.float32, device=device)

    xyza1_delta = torch.tensor([0.5, 0.3, 0.2, 0.01, 0.01, -0.02], dtype=pose1.dtype, device=device)
    xyza1_delta.requires_grad = True

    optimizer = torch.optim.Adam([{'params': xyza1_delta, 'lr': 0.01}])

    cloud2 = transform_cloud(cloud2, pose2)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2.detach().cpu())
    pcd2.paint_uniform_color([0, 0, 1])

    plt.figure(figsize=(20, 5))
    losses = []
    iters = []
    xyza_deltas = []
    # run optimization loop
    n_iters = 200
    for it in range(n_iters):
        # add noise to poses
        pose_deltas_mat = xyz_axis_angle_to_matrix(xyza1_delta[None]).squeeze()
        pose1_corr = torch.matmul(pose1, pose_deltas_mat)

        # transform point clouds to the same world coordinate frame
        cloud1_corr = transform_cloud(cloud1, pose1_corr)

        train_clouds = [cloud1_corr, cloud2]

        loss = point_to_point_dist(train_clouds, differentiable=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('At iter %i ICP loss: %f' % (it, loss))

        iters.append(it)
        losses.append(loss.item())
        xyza_deltas.append(xyza1_delta.clone())

        with torch.no_grad():
            plt.cla()
            plt.subplot(1, 3, 1)
            plt.ylabel('ICP point to point loss')
            plt.xlabel('Iterations')
            plt.plot(iters, losses, color='k')
            plt.grid(visible=True)

            plt.subplot(1, 3, 2)
            plt.ylabel('L2 pose distance')
            plt.xlabel('Iterations')
            plt.plot(iters, torch.stack(xyza_deltas, dim=0)[:, 0].cpu(), color='r', label='dx')
            plt.plot(iters, torch.stack(xyza_deltas, dim=0)[:, 1].cpu(), color='g', label='dy')
            plt.plot(iters, torch.stack(xyza_deltas, dim=0)[:, 2].cpu(), color='b', label='dz')
            plt.grid(visible=True)

            plt.subplot(1, 3, 3)
            plt.ylabel('L2 orient distance')
            plt.xlabel('Iterations')
            plt.plot(iters, torch.linalg.norm(torch.stack(xyza_deltas, dim=0)[:, 3:].cpu(), dim=1), label='da')
            plt.grid(visible=True)

            plt.pause(0.01)
            plt.draw()

            if it % 50 == 0 or it == n_iters - 1:
                print('Distance between clouds: %f', (torch.linalg.norm(pose1[:3, 3] - pose2[:3, 3])))
                print('Changed pose of the first cloud by: %s [m]' % torch.linalg.norm(xyza1_delta[:3]))

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(cloud1_corr.cpu())
                pcd1.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([pcd1, pcd2])
    plt.show()


def main():
    clouds_alignment_demo()


if __name__ == '__main__':
    main()
