# Differentiable Robotics

Collection of differentiable methods for robotics applications implemented with
[Pytorch](https://pytorch.org/).

Please, run the examples from the `src` directory of the repository.

## Path Smoothing

<img src="./docs/trajopt/path_smooth.gif" width="400">

```commandline
python -m trajopt.smooth_path
```

## ICP point cloud alignment

<img src="./docs/icp/clouds_before.png" width="400"> <img src="./docs/icp/clouds_aligned.png" width="400">
<img src="./docs/icp/icp_metrics.png" width="800">


```commandline
python -m icp.align_clouds
```

## Bounding box Pose Optimization

The optimization is based on a point cloud coverage function by the bounding box.

<img src="./docs/trajopt/bbox_pose_opt.gif" width="400">
