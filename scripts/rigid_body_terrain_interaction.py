import torch
import numpy as np
import matplotlib.pyplot as plt

# set matplotlib backend to show plots in a separate window
plt.switch_backend('Qt5Agg')
# set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# constants
g = 9.81
d_max = 6.4
grid_res = 0.1

# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def interpolate_height(z_grid, x_query, y_query, d_max, grid_res):
    """
    Interpolates the height at the desired (x_query, y_query) coordinates.

    Parameters:
    - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
    - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
    - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
    - d_max: Maximum distance from the origin.
    - grid_res: Grid resolution.

    Returns:
    - Interpolated z values at the queried coordinates.
    """

    # Ensure inputs are tensors
    z_grid = torch.as_tensor(z_grid)
    x_query = torch.as_tensor(x_query)
    y_query = torch.as_tensor(y_query)

    # Get the grid dimensions
    B, H, W = z_grid.shape

    # Flatten the grid coordinates
    z_grid_flat = z_grid.reshape(B, -1)

    # Flatten the query coordinates
    x_query_flat = x_query.reshape(B, -1)
    y_query_flat = y_query.reshape(B, -1)

    # Compute the indices of the grid points surrounding the query points
    x_i = torch.clamp(((x_query_flat + d_max) / grid_res).long(), 0, W - 2)
    y_i = torch.clamp(((y_query_flat + d_max) / grid_res).long(), 0, H - 2)

    # Compute the fractional part of the indices
    x_f = (x_query_flat + d_max) / grid_res - x_i.float()
    y_f = (y_query_flat + d_max) / grid_res - y_i.float()

    # Compute the indices of the grid points
    idx00 = x_i + W * y_i
    idx01 = x_i + W * (y_i + 1)
    idx10 = (x_i + 1) + W * y_i
    idx11 = (x_i + 1) + W * (y_i + 1)

    # Interpolate the z values
    z_query = (1 - x_f) * (1 - y_f) * z_grid_flat.gather(1, idx00) + \
              (1 - x_f) * y_f * z_grid_flat.gather(1, idx01) + \
              x_f * (1 - y_f) * z_grid_flat.gather(1, idx10) + \
              x_f * y_f * z_grid_flat.gather(1, idx11)

    return z_query


def integration_step(x, xd, dt, mode='rk4'):
    """
    Performs an integration step using the Euler method.

    Parameters:
    - x: Tensor of positions.
    - xd: Tensor of velocities.
    - dt: Time step.

    Returns:
    - Updated positions and velocities.
    """
    assert mode in ['euler', 'rk4']
    if mode == 'euler':
        x = x + xd * dt
    elif mode == 'rk4':
        k1 = dt * xd
        k2 = dt * (xd + k1 / 2)
        k3 = dt * (xd + k2 / 2)
        k4 = dt * (xd + k3)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def normailized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


# def surface_normals(x_grid, y_grid, z_grid, x_query, y_query):
#     """
#     Computes the surface normals and tangents at the queried coordinates.
#
#     Parameters:
#     - x_grid: Tensor of x coordinates (3D array), where the first dimension is the batch (B, H, W).
#     - y_grid: Tensor of y coordinates (3D array), where the first dimension is the batch (B, H, W).
#     - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
#     - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
#     - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
#
#     Returns:
#     - Surface normals and tangents at the queried coordinates.
#     """
#     # Interpolate the height at the queried coordinates
#     x_i = (x_query + d_max) / grid_res
#     y_i = (y_query + d_max) / grid_res
#     B, H, W = x_grid.shape
#     x_i = torch.clamp(x_i, 0, H - 2).long()
#     y_i = torch.clamp(y_i, 0, W - 2).long()
#     dz_dx = (z_grid[:, y_i, x_i + 1] - z_grid[:, y_i, x_i]) / grid_res
#     dz_dy = (z_grid[:, y_i + 1, x_i] - z_grid[:, y_i, x_i]) / grid_res
#
#     # n = [-dz_dx, -dz_dy, 1]
#     n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
#     n = normailized(n)
#
#     return n

def surface_normals(z_grid, x_query, y_query, d_max, grid_res):
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
    - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
    - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
    - d_max: Maximum distance from the origin.
    - grid_res: Grid resolution.

    Returns:
    - Surface normals and tangents at the queried coordinates.
    """
    # Ensure inputs are tensors
    z_grid = torch.as_tensor(z_grid)
    x_query = torch.as_tensor(x_query)
    y_query = torch.as_tensor(y_query)

    # Get the grid dimensions
    B, H, W = z_grid.shape

    # Compute the indices of the grid points surrounding the query points
    x_i = torch.clamp(((x_query + d_max) / grid_res).long(), 0, W - 2)
    y_i = torch.clamp(((y_query + d_max) / grid_res).long(), 0, H - 2)

    # Compute the fractional part of the indices
    x_f = (x_query + d_max) / grid_res - x_i.float()
    y_f = (y_query + d_max) / grid_res - y_i.float()

    # Compute the indices of the grid points
    idx00 = x_i + W * y_i
    idx01 = x_i + W * (y_i + 1)
    idx10 = (x_i + 1) + W * y_i
    idx11 = (x_i + 1) + W * (y_i + 1)

    # Interpolate the z values
    z_grid_flat = z_grid.reshape(B, -1)
    z00 = z_grid_flat.gather(1, idx00)
    z01 = z_grid_flat.gather(1, idx01)
    z10 = z_grid_flat.gather(1, idx10)
    z11 = z_grid_flat.gather(1, idx11)

    # Compute the surface normals
    dz_dx = (z10 - z00) * (1 - y_f) + (z11 - z01) * y_f
    dz_dy = (z01 - z00) * (1 - x_f) + (z11 - z10) * x_f
    n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
    n = normailized(n)

    return n

def rigid_body_params():
    """
    Returns the parameters of the rigid body.
    """
    # import open3d as o3d
    # robot = 'husky'
    # mesh_file = f'../data/meshes/{robot}.obj'
    # mesh = o3d.io.read_triangle_mesh(mesh_file)
    # n_points = 20
    # x_points = np.asarray(mesh.sample_points_uniformly(n_points).points)
    # x_points = torch.tensor(x_points, dtype=torch.float32)[None]

    size = (1.0, 0.5)
    s_x, s_y = size
    x_points = torch.stack([
        torch.hstack([torch.linspace(-s_x / 2., s_x / 2., 16 // 2), torch.linspace(-s_x / 2., s_x / 2., 16 // 2)]),
        torch.hstack([s_y / 2. * torch.ones(16 // 2), -s_y / 2. * torch.ones(16 // 2)]),
        torch.hstack([torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]),
                      torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2])])
    ]).T

    # divide the point cloud into left and right parts
    cog = x_points.mean(dim=0)
    mask_left = x_points[..., 1] > cog[1]
    mask_right = x_points[..., 1] < cog[1]

    m = torch.tensor(40.0)  # mass, kg
    I = inertia_tensor(m, x_points)
    I *= 100.0  # scale the inertia tensor as the point cloud is sparse

    return x_points, m, I, mask_left, mask_right


def heightmap(d_max, grid_res):
    x_grid = torch.arange(-d_max, d_max, grid_res)
    y_grid = torch.arange(-d_max, d_max, grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # z_grid = torch.zeros(x_grid.shape)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    z_grid = torch.exp(-(x_grid) ** 2 / 4) * torch.exp(-(y_grid-2) ** 2 / 2)

    return x_grid, y_grid, z_grid


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector.

    Parameters:
    - v: Input vector.

    Returns:
    - Skew-symmetric matrix of the input vector.
    """
    assert v.dim() == 2 and v.shape[1] == 3
    U = torch.zeros(v.shape[0], 3, 3, device=v.device)
    U[:, 0, 1] = -v[:, 2]
    U[:, 0, 2] = v[:, 1]
    U[:, 1, 2] = -v[:, 0]
    U[:, 1, 0] = v[:, 2]
    U[:, 2, 0] = -v[:, 1]
    U[:, 2, 1] = v[:, 0]
    return U


def forward_kinematics(x, xd, R, omega, x_points, xd_points,
                       z_grid, d_max, grid_res,
                       m, I_inv, mask_left, mask_right,
                       k_stiffness, k_damping, k_friction,
                       u_left, u_right):
    assert x.dim() == 2 and x.shape[1] == 3  # (B, 3)
    assert xd.dim() == 2 and xd.shape[1] == 3  # (B, 3)
    assert R.dim() == 3 and R.shape[-2:] == (3, 3)  # (B, 3, 3)
    assert x_points.dim() == 3 and x_points.shape[-1] == 3  # (B, N, 3)
    assert xd_points.dim() == 3 and xd_points.shape[-1] == 3  # (B, N, 3)
    assert mask_left.dim() == 2 and mask_left.shape[1] == x_points.shape[1]  # (B, N)
    assert mask_right.dim() == 2 and mask_right.shape[1] == x_points.shape[1]  # (B, N)
    assert u_left.dim() == 1  # (B,)
    assert u_right.dim() == 1  # (B,)
    assert z_grid.dim() == 3  # (B, H, W)
    assert I_inv.shape == (3, 3)  # (3, 3)
    B, n_pts, D = x_points.shape

    # check if the rigid body is in contact with the terrain
    z_points = interpolate_height(z_grid, x_points[..., 0], x_points[..., 1], d_max, grid_res)
    assert z_points.shape == (B, n_pts)
    dh_points = x_points[..., 2:3] - z_points.unsqueeze(-1)
    # in_contact = torch.sigmoid(-dh_points)
    in_contact = (dh_points <= 0.0).float()
    assert in_contact.shape == (B, n_pts, 1)

    # compute surface normals at the contact points
    n = surface_normals(z_grid, x_points[..., 0], x_points[..., 1], d_max, grid_res)
    assert n.shape == (B, n_pts, 3)

    # reaction at the contact points as spring-damper forces
    xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
    assert xd_points_n.shape == (B, n_pts, 1)
    F_spring = -torch.mul((k_stiffness * dh_points + k_damping * xd_points_n), n)  # F_s = -k * dh - b * v_n
    F_spring = torch.mul(F_spring, in_contact)
    assert F_spring.shape == (B, n_pts, 3)
    # limit the spring forces
    F_spring = torch.clamp(F_spring, min=0.0, max=2 * m * g)

    # friction forces: https://en.wikipedia.org/wiki/Friction
    N = torch.norm(F_spring, dim=-1, keepdim=True)
    xd_points_tau = xd_points - xd_points_n * n  # tangential velocities at the contact points
    tau = normailized(xd_points_tau)  # tangential directions of the velocities
    F_friction = -k_friction * N * tau  # F_fr = -k_fr * N * tau
    assert F_friction.shape == (B, n_pts, 3)

    # thrust forces: left and right
    thrust_dir = normailized(R @ torch.tensor([1.0, 0.0, 0.0]))
    x_left = x_points[mask_left].mean(dim=0, keepdims=True)  # left thrust is applied at the mean of the left points
    x_right = x_points[mask_right].mean(dim=0, keepdims=True)  # right thrust is applied at the mean of the right points
    F_thrust_left = u_left.unsqueeze(1) * thrust_dir * in_contact[mask_left].any()  # F_l = u_l * thrust_dir
    F_thrust_right = u_right.unsqueeze(1) * thrust_dir * in_contact[mask_right].any()  # F_r = u_r * thrust_dir
    assert F_thrust_left.shape == (B, 3) == F_thrust_right.shape
    torque_left = torch.cross(x_left - x, F_thrust_left)  # M_l = (x_l - x) x F_l
    torque_right = torch.cross(x_right - x, F_thrust_right)  # M_r = (x_r - x) x F_r
    torque_thrust = torque_left + torque_right  # M_thrust = M_l + M_r
    assert torque_thrust.shape == (B, 3)

    # rigid body rotation
    torque = torch.sum(torch.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1) + torque_thrust  # M = sum(r_i x F_i)
    omega_d = torque @ I_inv.transpose(0, 1)  # omega_d = I^(-1) M
    omega_skew = skew_symmetric(omega)  # omega_skew = [omega]_x
    dR = omega_skew @ R  # dR = [omega]_x R

    # motion of the cog
    F_grav = torch.tensor([[0.0, 0.0, -m * g]])
    F_cog = F_grav + F_spring.sum(dim=1) + F_friction.sum(dim=1) + F_thrust_left + F_thrust_right  # ma = sum(F_i)
    xdd = F_cog / m  # a = F / m
    assert xdd.shape == (B, 3)

    # motion of point composed of cog motion and rotation of the rigid body
    xd_points = xd.unsqueeze(1) + torch.cross(omega.view(B, 1, 3), x_points - x.unsqueeze(1))  # Koenig's theorem in mechanics
    assert xd_points.shape == (B, n_pts, 3)

    return xd, xdd, dR, omega_d, xd_points, F_spring, F_friction, F_thrust_left, F_thrust_right


def update_states(x, xd, xdd, R, dR, omega, omega_d, x_points, xd_points, dt):
    xd = integration_step(xd, xdd, dt)
    x = integration_step(x, xd, dt)
    x_points = integration_step(x_points, xd_points, dt)
    omega = integration_step(omega, omega_d, dt)
    R = integration_step(R, dR, dt)

    return x, xd, R, omega, x_points

def dphysics(state, xd_points,
             z_grid, d_max, grid_res,
             m, I, mask_left, mask_right, controls,
             k_stiffness=1000., k_damping=None, k_friction=0.5,
             T=10.0, dt=0.01):
    # state: x, xd, R, omega, x_points
    x, xd, R, omega, x_points = state

    I_inv = torch.inverse(I)
    if k_damping is None:
        k_damping = np.sqrt(4 * m * k_stiffness)  # critically damping

    # dynamics of the rigid body
    Xs, Xds, Rs, Omegas, Omega_ds, X_points = [], [], [], [], [], []
    F_springs, F_frictions, F_thrusts_left, F_thrusts_right = [], [], [], []
    ts = range(int(T / dt))
    B, N_ts, N_pts = x.shape[0], len(ts), x_points.shape[1]
    for i in ts:
        # control inputs
        u_left, u_right = controls[:, i, 0], controls[:, i, 1]  # thrust forces, Newtons or kg*m/s^2
        # forward kinematics
        (xd, xdd, dR, omega_d, xd_points,
         F_spring, F_friction, F_thrust_left, F_thrust_right) = forward_kinematics(x, xd, R, omega, x_points, xd_points,
                                                                                   z_grid, d_max, grid_res,
                                                                                   m, I_inv, mask_left, mask_right,
                                                                                   k_stiffness, k_damping, k_friction,
                                                                                   u_left, u_right)
        # update states: integration steps
        x, xd, R, omega, x_points = update_states(x, xd, xdd, R, dR, omega, omega_d, x_points, xd_points, dt)

        # save states
        Xs.append(x)
        Xds.append(xd)
        Rs.append(R)
        Omegas.append(omega)
        X_points.append(x_points)

        # save forces
        F_springs.append(F_spring)
        F_frictions.append(F_friction)
        F_thrusts_left.append(F_thrust_left)
        F_thrusts_right.append(F_thrust_right)

    # to tensors
    Xs = torch.stack(Xs).transpose(1, 0)
    assert Xs.shape == (B, N_ts, 3)
    Xds = torch.stack(Xds).transpose(1, 0)
    assert Xds.shape == (B, N_ts, 3)
    Rs = torch.stack(Rs).transpose(1, 0)
    assert Rs.shape == (B, N_ts, 3, 3)
    Omegas = torch.stack(Omegas).transpose(1, 0)
    assert Omegas.shape == (B, N_ts, 3)
    X_points = torch.stack(X_points).transpose(1, 0)
    assert X_points.shape == (B, N_ts, N_pts, 3)
    F_springs = torch.stack(F_springs).transpose(1, 0)
    assert F_springs.shape == (B, N_ts, N_pts, 3)
    F_frictions = torch.stack(F_frictions).transpose(1, 0)
    assert F_frictions.shape == (B, N_ts, N_pts, 3)
    F_thrusts_left = torch.stack(F_thrusts_left).transpose(1, 0)
    assert F_thrusts_left.shape == (B, N_ts, 3)
    F_thrusts_right = torch.stack(F_thrusts_right).transpose(1, 0)
    assert F_thrusts_right.shape == (B, N_ts, 3)

    return Xs, Xds, Rs, Omegas, X_points, F_springs, F_frictions, F_thrusts_left, F_thrusts_right


def inertia_tensor(mass, points):
    """
    Compute the inertia tensor for a rigid body represented by point masses.

    Parameters:
    mass (float): The total mass of the body.
    points (array-like): A list or array of points (x, y, z) representing the mass distribution.
                         Each point contributes equally to the total mass.

    Returns:
    torch.Tensor: A 3x3 inertia tensor matrix.
    """

    # Convert points to a tensor
    points = torch.as_tensor(points)

    # Number of points
    n_points = points.shape[0]

    # Mass per point: assume uniform mass distribution
    mass_per_point = mass / n_points

    # Initialize the inertia tensor components
    Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0

    # Loop over each point and accumulate the inertia tensor components
    for x, y, z in points:
        Ixx += mass_per_point * (y ** 2 + z ** 2)
        Iyy += mass_per_point * (x ** 2 + z ** 2)
        Izz += mass_per_point * (x ** 2 + y ** 2)
        Ixy -= mass_per_point * x * y
        Ixz -= mass_per_point * x * z
        Iyz -= mass_per_point * y * z

    # Construct the inertia tensor matrix
    I = torch.tensor([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])

    return I

def motion():
    # simulation parameters
    dt = 0.01
    T = 5.0

    # control inputs in Newtons
    controls = torch.stack([torch.tensor([[110.0, 110.0]] * int(T / dt)), torch.tensor([[110.0, 100.0]] * int(T / dt))])

    # rigid body parameters
    x_points, m, I, mask_left, mask_right = rigid_body_params()

    # initial state
    x = torch.tensor([[-2.0, 0.0, 1.0], [-2.5, 1.0, 1.0]])
    xd = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    R = torch.eye(3).repeat(x.shape[0], 1, 1)
    omega = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
    xd_points = torch.zeros_like(x_points)
    mask_left = mask_left.repeat(x.shape[0], 1)
    mask_right = mask_right.repeat(x.shape[0], 1)

    # heightmap defining the terrain
    x_grid, y_grid, z_grid = heightmap(d_max, grid_res)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid = z_grid.repeat(x.shape[0], 1, 1)

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # simulate the rigid body dynamics
    (Xs, Xds, Rs, Omegas, X_points,
     F_springs, F_frictions, F_thrusts_left, F_thrusts_right) = dphysics(state0, xd_points,
                                                                         z_grid, d_max, grid_res,
                                                                         m, I, mask_left, mask_right,
                                                                         controls,
                                                                         T=T, dt=dt)
    # visualize
    for batch_i in range(Xs.shape[0]):
        states = (Xs[batch_i], Xds[batch_i], Rs[batch_i], Omegas[batch_i], X_points[batch_i])
        forces = (F_springs[batch_i], F_frictions[batch_i], F_thrusts_left[batch_i], F_thrusts_right[batch_i])
        visualize_traj(states,
                       x_grid[batch_i], y_grid[batch_i], z_grid[batch_i],
                       forces=forces,
                       vis_step=10,
                       mask_left=mask_left[batch_i], mask_right=mask_right[batch_i])
    

def visualize_traj(states, x_grid, y_grid, z_grid, forces=None, vis_step=1, mask_left=None, mask_right=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    xs, xds, Rs, omegas, X_points = states
    N_ts, N_pts = X_points.shape[0], X_points.shape[1]
    assert xs.shape == (N_ts, 3)
    assert xds.shape == (N_ts, 3)
    assert Rs.shape == (N_ts, 3, 3)
    assert omegas.shape == (N_ts, 3)
    assert X_points.shape == (N_ts, N_pts, 3)
    with torch.no_grad():
        for i, (x, xd, R, omega, x_points) in enumerate(zip(xs, xds, Rs, omegas, X_points)):
            if i % vis_step != 0:
                continue
            assert x.shape == (3,)
            assert xd.shape == (3,)
            assert R.shape == (3, 3)
            assert omega.shape == (3,)
            assert x_points.shape == (N_pts, 3)

            plt.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # plot rigid body points and cog
            ax.scatter(x_points[:, 0].cpu(), x_points[:, 1].cpu(), x_points[:, 2].cpu(), c='k')
            ax.scatter(x[0].item(), x[1].item(), x[2].item(), c='r')

            # plot rigid body frame
            x_axis = R @ torch.tensor([1.0, 0.0, 0.0])
            y_axis = R @ torch.tensor([0.0, 1.0, 0.0])
            z_axis = R @ torch.tensor([0.0, 0.0, 1.0])
            ax.quiver(x[0], x[1], x[2], x_axis[0], x_axis[1], x_axis[2], color='r')
            ax.quiver(x[0], x[1], x[2], y_axis[0], y_axis[1], y_axis[2], color='g')
            ax.quiver(x[0], x[1], x[2], z_axis[0], z_axis[1], z_axis[2], color='b')

            # plot trajectory
            ax.plot(xs[:, 0].cpu(), xs[:, 1].cpu(), xs[:, 2].cpu(), c='b')

            # plot cog velocity
            ax.quiver(x[0], x[1], x[2], xd[0], xd[1], xd[2], color='k')

            # plot terrain: somehow the height map is flipped, need to transpose it
            ax.plot_surface(x_grid.cpu(), y_grid.cpu(), z_grid.cpu().T, alpha=0.5, cmap='terrain')
            
            # plot forces
            if forces is not None:
                F_springs, F_frictions, F_thrusts_left, F_thrusts_right = forces
                F_spring, F_friction, F_thrust_left, F_thrust_right = F_springs[i], F_frictions[i], F_thrusts_left[i], F_thrusts_right[i]
                # plot normal forces
                # ax.quiver(x_points[:, 0], x_points[:, 1], x_points[:, 2], F_spring[:, 0], F_spring[:, 1], F_spring[:, 2], color='b')
                F_spring_total = F_spring.sum(dim=0)
                ax.quiver(x[0], x[1], x[2], F_spring_total[0] / g, F_spring_total[1] / g, F_spring_total[2] / g, color='b')

                # plot friction forces
                # ax.quiver(x_points[:, 0], x_points[:, 1], x_points[:, 2], F_friction[:, 0], F_friction[:, 1], F_friction[:, 2], color='g')
                F_friction_total = F_friction.sum(dim=0)
                ax.quiver(x[0], x[1], x[2], F_friction_total[0] / g, F_friction_total[1] / g, F_friction_total[2] / g, color='g')

                # plot thrust forces
                if mask_left is not None and mask_right is not None:
                    ax.quiver(x_points[mask_left].mean(dim=0)[0], x_points[mask_left].mean(dim=0)[1], x_points[mask_left].mean(dim=0)[2],
                              F_thrust_left[0] / g, F_thrust_left[1] / g, F_thrust_left[2] / g, color='r')
                    ax.quiver(x_points[mask_right].mean(dim=0)[0], x_points[mask_right].mean(dim=0)[1], x_points[mask_right].mean(dim=0)[2],
                              F_thrust_right[0] / g, F_thrust_right[1] / g, F_thrust_right[2] / g, color='r')
                else:
                    F_thrust_total = F_thrust_left + F_thrust_right
                    ax.quiver(x[0], x[1], x[2], F_thrust_total[0] / g, F_thrust_total[1] / g, F_thrust_total[2] / g, color='r')

            set_axes_equal(ax)
            plt.pause(0.01)

        plt.show()


def optimization():
    # simulation parameters
    dt = 0.01
    T = 5.0
    vis = True
    n_iters = 100
    lr = 0.002

    # rigid body parameters
    x_points, m, I, mask_left, mask_right = rigid_body_params()

    # initial state
    x = torch.tensor([
        [-2.0, 0.0, 1.0]
    ])
    xd = torch.tensor([
        [0.0, 0.0, 0.0]
    ])
    R = torch.eye(3).repeat(x.shape[0], 1, 1)
    omega = torch.tensor([
        [0.0, 0.0, 0.0]
    ])
    x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
    xd_points = torch.zeros_like(x_points)
    mask_left = mask_left.repeat(x.shape[0], 1)
    mask_right = mask_right.repeat(x.shape[0], 1)

    # heightmap defining the terrain
    x_grid, y_grid, z_grid_gt = heightmap(d_max, grid_res)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid_gt = z_grid_gt.repeat(x.shape[0], 1, 1)

    # control inputs in Newtons
    controls = torch.tensor([
        [[110.0, 110.0]] * int(T / dt)
    ])

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # simulate the rigid body dynamics
    (Xs_gt, Xds_gt, Rs_gt, Omegas_gt, X_points_gt,
     F_springs_gt, F_frictions_gt, F_thrusts_left_gt, F_thrusts_right_gt) = dphysics(state0, xd_points,
                                                                                     z_grid_gt, d_max, grid_res,
                                                                                     m, I, mask_left, mask_right,
                                                                                     controls,
                                                                                     T=T, dt=dt)
    if vis:
        for batch_i in range(len(Xs_gt)):
            states_gt = (Xs_gt[batch_i], Xds_gt[batch_i], Rs_gt[batch_i], Omegas_gt[batch_i], X_points_gt[batch_i])
            forces_gt = (F_springs_gt[batch_i], F_frictions_gt[batch_i], F_thrusts_left_gt[batch_i], F_thrusts_right_gt[batch_i])
            visualize_traj(states_gt, x_grid[batch_i], y_grid[batch_i], z_grid_gt[batch_i], forces_gt, vis_step=10)

    # initial guess for the heightmap
    z_grid = torch.zeros_like(z_grid_gt, requires_grad=True)

    # optimization
    optimizer = torch.optim.Adam([z_grid], lr=lr)
    z_grid_best = z_grid.clone()
    loss_best = np.inf
    for i in range(n_iters):
        optimizer.zero_grad()
        # simulate the rigid body dynamics
        (Xs, Xds, Rs, Omegas, X_points,
         F_springs, F_frictions, F_thrusts_left, F_thrusts_right) = dphysics(state0, xd_points,
                                                                             z_grid, d_max, grid_res,
                                                                             m, I, mask_left, mask_right,
                                                                             controls,
                                                                             T=T, dt=dt)

        # compute the loss
        loss_x = torch.nn.functional.mse_loss(Xs, Xs_gt)
        loss_xd = torch.nn.functional.mse_loss(Xds, Xds_gt)
        loss = loss_x + loss_xd
        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss x: {loss_x.item():.3f}, Loss xd: {loss_xd.item():.3f}')

        if loss.item() < loss_best:
            loss_best = loss.item()
            z_grid_best = z_grid.clone()

        # heightmap difference
        with torch.no_grad():
            z_diff = torch.nn.functional.mse_loss(z_grid, z_grid_gt)
            print(f'Heightmap difference: {z_diff.item()}')

        if vis and (i == 0 or i == n_iters - 1):
            for batch_i in range(len(Xs)):
                states = (Xs[batch_i], Xds[batch_i], Rs[batch_i], Omegas[batch_i], X_points[batch_i])
                forces = (F_springs[batch_i], F_frictions[batch_i], F_thrusts_left[batch_i], F_thrusts_right[batch_i])
                visualize_traj(states, x_grid[batch_i], y_grid[batch_i], z_grid_best[batch_i], forces, vis_step=10)


def main():
    motion()
    # optimization()


if __name__ == '__main__':
    main()
