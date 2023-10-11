import numpy as np


__all__ = [
    'read_poses',
    'read_cloud'
]


def read_poses(path):
    poses = np.genfromtxt(path, delimiter=', ', skip_header=True)
    ids = np.genfromtxt(path, delimiter=', ', dtype=str, skip_header=True)[:, 0].tolist()
    # assert ids == list(range(len(ids)))
    poses = poses[:, 2:]
    poses = poses.reshape((-1, 4, 4))
    poses = dict(zip(ids, poses))
    return poses


def read_cloud(npz_file):
    cloud = np.load(npz_file)['cloud']
    if cloud.ndim == 2:
        cloud = cloud.reshape((-1,))
    return cloud
