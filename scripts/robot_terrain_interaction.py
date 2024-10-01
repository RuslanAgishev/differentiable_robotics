import torch
import numpy as np
import os
import matplotlib.pyplot as plt


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


class DPhysConfig:
    def __init__(self):
        # robot parameters
        self.robot_mass = 40.  # kg
        self.robot_size = (1.0, 0.5)  # length, width in meters
        self.robot_points, self.robot_mask_left, self.robot_mask_right = self.rigid_body_geometry(from_mesh=False)
        self.robot_I = inertia_tensor(self.robot_mass, self.robot_points)  # 3x3 inertia tensor, kg*m^2
        self.robot_I *= 10.  # increase inertia for stability, as the point cloud is very sparse
        self.vel_max = 1.2  # m/s
        self.omega_max = 0.4  # rad/s

        # height map parameters
        self.grid_res = 0.1
        self.d_max = 6.4
        self.k_stiffness = 5_000.
        self.k_damping = float(np.sqrt(4 * self.robot_mass * self.k_stiffness))  # critical damping
        self.k_friction = 0.5

        # trajectory shooting parameters
        self.traj_sim_time = 5.0
        self.dt = 0.01
        self.n_sim_trajs = 32
        self.integration_mode = 'rk4'  # 'euler', 'rk2', 'rk4'

    def rigid_body_geometry(self, from_mesh=False):
        """
        Returns the parameters of the rigid body.
        """
        if from_mesh:
            import open3d as o3d
            robot = 'tradr'
            mesh_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/meshes/{robot}.obj')
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            n_points = 32
            x_points = np.asarray(mesh.sample_points_uniformly(n_points).points, dtype=np.float32)
            x_points = torch.as_tensor(x_points)
        else:
            size = self.robot_size
            s_x, s_y = size
            x_points = torch.stack([
                torch.hstack([torch.linspace(-s_x / 2., s_x / 2., 16 // 2),
                              torch.linspace(-s_x / 2., s_x / 2., 16 // 2)]),
                torch.hstack([s_y / 2. * torch.ones(16 // 2),
                              -s_y / 2. * torch.ones(16 // 2)]),
                torch.hstack([torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]),
                              torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2])])
            ]).T

        # divide the point cloud into left and right parts
        cog = x_points.mean(dim=0)
        mask_left = x_points[..., 1] > cog[1]
        mask_right = x_points[..., 1] < cog[1]

        return x_points, mask_left, mask_right

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

def vw_to_track_vel(v, w, r=1.0):
    # v: linear velocity, w: angular velocity, r: robot radius
    # v = (v_l + v_r) / 2
    # w = (v_l - v_r) / (2 * r)
    v_l = v + r * w
    v_r = v - r * w
    return v_l, v_r

class DPhysics(torch.nn.Module):
    def __init__(self, dphys_cfg=DPhysConfig(), device='cpu'):
        super(DPhysics, self).__init__()
        self.dphys_cfg = dphys_cfg
        self.device = device
        self.I = torch.as_tensor(dphys_cfg.robot_I, device=device)
        self.I_inv = torch.inverse(self.I)
        self.g = 9.81  # gravity, m/s^2
        self.robot_mask_left = torch.as_tensor(dphys_cfg.robot_mask_left, device=device)
        self.robot_mask_right = torch.as_tensor(dphys_cfg.robot_mask_right, device=device)

    def forward_kinematics(self, state, xd_points,
                           z_grid, stiffness, damping, friction,
                           m, mask_left, mask_right,
                           u_left, u_right):
        # unpack state
        x, xd, R, omega, x_points = state
        assert x.dim() == 2 and x.shape[1] == 3  # (B, 3)
        assert xd.dim() == 2 and xd.shape[1] == 3  # (B, 3)
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)  # (B, 3, 3)
        assert x_points.dim() == 3 and x_points.shape[-1] == 3  # (B, N, 3)
        assert xd_points.dim() == 3 and xd_points.shape[-1] == 3  # (B, N, 3)
        assert mask_left.dim() == 1 and mask_left.shape[0] == x_points.shape[1]  # (N,)
        assert mask_right.dim() == 1 and mask_right.shape[0] == x_points.shape[1]  # (N,)
        # if scalar, convert to tensor
        if isinstance(u_left, (int, float)):
            u_left = torch.tensor([u_left], device=self.device)
        if isinstance(u_right, (int, float)):
            u_right = torch.tensor([u_right], device=self.device)
        assert u_left.dim() == 1  # scalar
        assert u_right.dim() == 1  # scalar
        assert z_grid.dim() == 3  # (B, H, W)
        B, n_pts, D = x_points.shape

        # compute the terrain properties at the robot points
        z_points = self.interpolate_grid(z_grid, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
        assert z_points.shape == (B, n_pts, 1)
        if not isinstance(stiffness, (int, float)):
            stiffness_points = self.interpolate_grid(stiffness, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
            assert stiffness_points.shape == (B, n_pts, 1)
        else:
            stiffness_points = stiffness
        if not isinstance(damping, (int, float)):
            damping_points = self.interpolate_grid(damping, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
            assert damping_points.shape == (B, n_pts, 1)
        else:
            damping_points = damping
        if not isinstance(friction, (int, float)):
            friction_points = self.interpolate_grid(friction, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
            assert friction_points.shape == (B, n_pts, 1)
        else:
            friction_points = friction

        # check if the rigid body is in contact with the terrain
        dh_points = x_points[..., 2:3] - z_points
        on_grid = (x_points[..., 0:1] >= -self.dphys_cfg.d_max) & (x_points[..., 0:1] <= self.dphys_cfg.d_max) & \
                    (x_points[..., 1:2] >= -self.dphys_cfg.d_max) & (x_points[..., 1:2] <= self.dphys_cfg.d_max)
        in_contact = ((dh_points <= 0.0) & on_grid).float()
        assert in_contact.shape == (B, n_pts, 1)

        # compute surface normals at the contact points
        n = self.surface_normals(z_grid, x_points[..., 0], x_points[..., 1])
        assert n.shape == (B, n_pts, 3)

        # reaction at the contact points as spring-damper forces
        xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
        assert xd_points_n.shape == (B, n_pts, 1)
        F_spring = -torch.mul((stiffness_points * dh_points + damping_points * xd_points_n), n)  # F_s = -k * dh - b * v_n
        F_spring = torch.mul(F_spring, in_contact)
        assert F_spring.shape == (B, n_pts, 3)

        # friction forces: https://en.wikipedia.org/wiki/Friction
        thrust_dir = normailized(R @ torch.tensor([1.0, 0.0, 0.0], device=self.device))
        N = torch.norm(F_spring, dim=2)  # normal force magnitude at the contact points
        v_l = (u_left.unsqueeze(1) * thrust_dir).unsqueeze(1)  # left track velocity
        v_r = (u_right.unsqueeze(1) * thrust_dir).unsqueeze(1)  # right track velocity
        F_friction = torch.zeros_like(F_spring)  # initialize friction forces
        # F_fr = -mu * N * tanh(v_cmd - xd_points)  # left and right track friction forces
        F_friction[:, mask_left] = (friction_points * N.unsqueeze(2) * torch.tanh(v_l - xd_points))[:, mask_left]
        F_friction[:, mask_right] = (friction_points * N.unsqueeze(2) * torch.tanh(v_r - xd_points))[:, mask_right]
        assert F_friction.shape == (B, n_pts, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1)
        omega_d = torque @ self.I_inv.transpose(0, 1)  # omega_d = I^(-1) M
        Omega_skew = skew_symmetric(omega)  # Omega_skew = [omega]_x
        dR = Omega_skew @ R  # dR = [omega]_x R

        # motion of the cog
        F_grav = torch.tensor([[0.0, 0.0, -m * self.g]], device=self.device)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_spring.mean(dim=1) + F_friction.mean(dim=1)  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        # motion of point composed of cog motion and rotation of the rigid body (Koenig's theorem in mechanics)
        xd_points = xd.unsqueeze(1) + torch.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))
        assert xd_points.shape == (B, n_pts, 3)

        dstate = (xd, xdd, dR, omega_d, xd_points)
        forces = (F_spring, F_friction)

        return dstate, forces

    def update_state(self, state, dstate, dt):
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, omega, x_points = state
        _, xdd, dR, omega_d, xd_points = dstate

        xd = self.integration_step(xd, xdd, dt, mode=self.dphys_cfg.integration_mode)
        x = self.integration_step(x, xd, dt, mode=self.dphys_cfg.integration_mode)
        x_points = self.integration_step(x_points, xd_points, dt, mode=self.dphys_cfg.integration_mode)
        omega = self.integration_step(omega, omega_d, dt, mode=self.dphys_cfg.integration_mode)
        # R = self.integration_step(R, dR, dt, mode=self.dphys_cfg.integration_mode)
        R = self.integrate_rotation(R, omega, dt)

        state = (x, xd, R, omega, x_points)

        return state

    @staticmethod
    def integrate_rotation(R, omega, dt, eps=1e-6):
        """
        Integrates the rotation matrix for the next time step using Rodrigues' formula.

        Parameters:
        - R: Tensor of rotation matrices.
        - omega: Tensor of angular velocities.
        - dt: Time step.
        - eps: Small value to avoid division by zero.

        Returns:
        - Updated rotation matrices.

        Reference:
            https://math.stackexchange.com/questions/167880/calculating-new-rotation-matrix-with-its-derivative-given
        """
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)
        assert omega.dim() == 2 and omega.shape[1] == 3
        assert dt > 0

        # Compute the skew-symmetric matrix of the angular velocities
        Omega_x = skew_symmetric(omega)

        # Compute exponential map of the skew-symmetric matrix
        theta = torch.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)

        # Normalize the angular velocities
        Omega_x_norm = Omega_x / (theta + eps)

        # Rodrigues' formula: R_new = R * (I + Omega_x * sin(theta * dt) + Omega_x^2 * (1 - cos(theta * dt)))
        I = torch.eye(3).to(R.device)
        R_new = R @ (I + Omega_x_norm * torch.sin(theta * dt) + Omega_x_norm @ Omega_x_norm * (1 - torch.cos(theta * dt)))

        return R_new

    @staticmethod
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
        if mode == 'euler':
            x = x + xd * dt
        elif mode == 'rk2':
            k1 = dt * xd
            k2 = dt * (xd + k1)
            x = x + k2 / 2
        elif mode == 'rk4':
            k1 = dt * xd
            k2 = dt * (xd + k1 / 2)
            k3 = dt * (xd + k2 / 2)
            k4 = dt * (xd + k3)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(f'Unknown integration mode: {mode}')
        return x

    def surface_normals(self, z_grid, x_query, y_query):
        """
        Computes the surface normals and tangents at the queried coordinates.

        Parameters:
        - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
        - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
        - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).

        Returns:
        - Surface normals at the queried coordinates.
        """
        # unpack config
        d_max = self.dphys_cfg.d_max
        grid_res = self.dphys_cfg.grid_res

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
        n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)  # n = [-dz/dx, -dz/dy, 1]
        n = normailized(n)

        return n

    def interpolate_grid(self, grid, x_query, y_query):
        """
        Interpolates the height at the desired (x_query, y_query) coordinates.

        Parameters:
        - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, H, W).
        - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
        - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).

        Returns:
        - Interpolated grid values at the queried coordinates.
        """
        # unpack config
        d_max = self.dphys_cfg.d_max
        grid_res = self.dphys_cfg.grid_res

        # Ensure inputs are tensors
        grid = torch.as_tensor(grid)
        x_query = torch.as_tensor(x_query)
        y_query = torch.as_tensor(y_query)

        # Get the grid dimensions
        B, H, W = grid.shape

        # Flatten the grid coordinates
        z_grid_flat = grid.reshape(B, -1)

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

        # Interpolate the z values (linear interpolation)
        z_query = (1 - x_f) * (1 - y_f) * z_grid_flat.gather(1, idx00) + \
                  (1 - x_f) * y_f * z_grid_flat.gather(1, idx01) + \
                  x_f * (1 - y_f) * z_grid_flat.gather(1, idx10) + \
                  x_f * y_f * z_grid_flat.gather(1, idx11)

        return z_query

    def dphysics(self, z_grid, controls, state=None, stiffness=None, damping=None, friction=None):
        """
        Simulates the dynamics of the robot moving on the terrain.

        Parameters:
        - z_grid: Tensor of the height map (B, H, W).
        - controls: Tensor of control inputs (B, N, 2).
        - state: Tuple of the robot state (x, xd, R, omega, x_points).
        - stiffness: scalar or Tensor of the stiffness values at the robot points (B, H, W).
        - damping: scalar or Tensor of the damping values at the robot points (B, H, W).
        - friction: scalar or Tensor of the friction values at the robot points (B, H, W).

        Returns:
        - Tuple of the robot states and forces:
            - states: Tuple of the robot states (x, xd, R, omega, x_points).
            - forces: Tuple of the forces (F_springs, F_frictions, F_thrusts_left, F_thrusts_right).
        """
        # unpack config
        device = self.device
        dt = self.dphys_cfg.dt
        T = self.dphys_cfg.traj_sim_time
        batch_size = z_grid.shape[0]

        # robot geometry masks for left and right thrust points
        mask_left = self.robot_mask_left
        mask_right = self.robot_mask_right

        # initial state
        if state is None:
            x = torch.tensor([0.0, 0.0, 0.2]).to(device).repeat(batch_size, 1)
            xd = torch.zeros_like(x)
            R = torch.eye(3).to(device).repeat(batch_size, 1, 1)
            omega = torch.zeros_like(x)
            x_points = torch.as_tensor(self.dphys_cfg.robot_points, device=device)
            x_points = x_points.repeat(batch_size, 1, 1)
            x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
            state = (x, xd, R, omega, x_points)

        # terrain properties
        stiffness = self.dphys_cfg.k_stiffness if stiffness is None else stiffness
        damping = self.dphys_cfg.k_damping if damping is None else damping
        friction = self.dphys_cfg.k_friction if friction is None else friction

        N_ts = min(int(T / dt), controls.shape[1])
        B = state[0].shape[0]
        assert controls.shape == (B, N_ts, 2)  # for each time step, left and right thrust forces

        # TODO: there is some bug, had to transpose grid map
        z_grid = z_grid.transpose(1, 2)  # (B, H, W) -> (B, W, H)
        stiffness = stiffness.transpose(1, 2) if not isinstance(stiffness, (int, float)) else stiffness
        damping = damping.transpose(1, 2) if not isinstance(damping, (int, float)) else damping
        friction = friction.transpose(1, 2) if not isinstance(friction, (int, float)) else friction

        # state: x, xd, R, omega, x_points
        x, xd, R, omega, x_points = state
        xd_points = torch.zeros_like(x_points)

        # dynamics of the rigid body
        Xs, Xds, Rs, Omegas, Omega_ds, X_points = [], [], [], [], [], []
        F_springs, F_frictions = [], []
        ts = range(N_ts)
        B, N_ts, N_pts = x.shape[0], len(ts), x_points.shape[1]
        for t in ts:
            # control inputs
            u_left, u_right = controls[:, t, 0], controls[:, t, 1]  # thrust forces, Newtons or kg*m/s^2
            # forward kinematics
            dstate, forces = self.forward_kinematics(state=state, xd_points=xd_points,
                                                     z_grid=z_grid,
                                                     stiffness=stiffness, damping=damping, friction=friction,
                                                     m=self.dphys_cfg.robot_mass,
                                                     mask_left=mask_left, mask_right=mask_right,
                                                     u_left=u_left, u_right=u_right,)
            # update state: integration steps
            state = self.update_state(state, dstate, dt)

            # unpack state, its differential, and forces
            x, xd, R, omega, x_points = state
            _, xdd, dR, omega_d, xd_points = dstate
            F_spring, F_friction = forces

            # save states
            Xs.append(x)
            Xds.append(xd)
            Rs.append(R)
            Omegas.append(omega)
            X_points.append(x_points)

            # save forces
            F_springs.append(F_spring)
            F_frictions.append(F_friction)

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

        States = Xs, Xds, Rs, Omegas, X_points
        Forces = F_springs, F_frictions

        return States, Forces

    def forward(self, z_grid, controls, state=None, **kwargs):
        return self.dphysics(z_grid, controls, state, **kwargs)


def visualize_traj(states, x_grid, y_grid, z_grid, forces, vis_step=1):
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
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    xs, xds, Rs, omegas, X_points = states
    F_springs, F_frictions = forces
    N_ts, N_pts = X_points.shape[0], X_points.shape[1]
    assert xs.shape == (N_ts, 3)
    assert xds.shape == (N_ts, 3)
    assert Rs.shape == (N_ts, 3, 3)
    assert omegas.shape == (N_ts, 3)
    assert X_points.shape == (N_ts, N_pts, 3)
    assert F_springs.shape == (N_ts, N_pts, 3)
    assert F_frictions.shape == (N_ts, N_pts, 3)
    with torch.no_grad():
        for i, (x, xd, R, omega, x_points, f_spring, f_friction) in enumerate(zip(xs, xds, Rs, omegas, X_points, F_springs, F_frictions)):
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
            ax.scatter(x_points[:, 0], x_points[:, 1], x_points[:, 2], c='k')
            ax.scatter(x[0].item(), x[1].item(), x[2].item(), c='r')

            # plot rigid body frame
            x_axis = R @ np.array([1.0, 0.0, 0.0])
            y_axis = R @ np.array([0.0, 1.0, 0.0])
            z_axis = R @ np.array([0.0, 0.0, 1.0])
            ax.quiver(x[0], x[1], x[2], x_axis[0], x_axis[1], x_axis[2], color='r')
            ax.quiver(x[0], x[1], x[2], y_axis[0], y_axis[1], y_axis[2], color='g')
            ax.quiver(x[0], x[1], x[2], z_axis[0], z_axis[1], z_axis[2], color='b')

            # plot trajectory
            ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], c='b')

            # plot cog velocity
            ax.quiver(x[0], x[1], x[2], xd[0], xd[1], xd[2], color='k')

            # plot terrain: somehow the height map is flipped, need to transpose it
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='terrain')

            # plot forces: spring and friction
            f_spring /= 9.81  # normalize the spring forces
            f_friction /= 9.81
            ax.quiver(x_points[:, 0], x_points[:, 1], x_points[:, 2],
                      f_spring[:, 0], f_spring[:, 1], f_spring[:, 2], color='b')
            ax.quiver(x_points[:, 0], x_points[:, 1], x_points[:, 2],
                      f_friction[:, 0], f_friction[:, 1], f_friction[:, 2], color='g')

            set_axes_equal(ax)
            plt.pause(0.01)

        plt.show()


def motion():
    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use('Qt5Agg')

    # rigid body and terrain parameters
    dphys_cfg = DPhysConfig()
    device = torch.device('cpu')

    # simulation parameters
    dt = dphys_cfg.dt
    dphys_cfg.traj_sim_time = 5.0
    dphys_cfg.k_friction = 1.0
    T = dphys_cfg.traj_sim_time

    # control inputs: [vel_left, vel_right] in m/s
    controls = torch.stack([
        torch.tensor([[0.5, 0.6]] * int(T / dt)),
    ]).to(device)
    B, N_ts = controls.shape[:2]
    assert controls.shape == (B, N_ts, 2)

    # initial state
    x = torch.stack([
        torch.tensor([0.0, 0.0, 0.2]),
    ]).to(device)
    assert x.shape == (B, 3)
    xd = torch.zeros_like(x)
    assert xd.shape == (B, 3)
    R = torch.tensor(Rotation.from_euler('z', [0.0]).as_matrix(), dtype=torch.float32, device=device)
    assert R.shape == (B, 3, 3)
    omega = torch.zeros_like(x)
    assert omega.shape == (B, 3)
    x_points = torch.as_tensor(dphys_cfg.robot_points, device=device).repeat(x.shape[0], 1, 1)
    assert x_points.shape == (B, len(dphys_cfg.robot_points), 3)
    x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
    assert x_points.shape == (B, len(dphys_cfg.robot_points), 3)
    state0 = (x, xd, R, omega, x_points)

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # z_grid = (torch.sin(x_grid) * torch.cos(y_grid)).to(device)
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2).to(device)
    # z_grid = torch.zeros_like(x_grid)
    stiffness = dphys_cfg.k_stiffness * torch.ones_like(z_grid)
    friction = dphys_cfg.k_friction * torch.ones_like(z_grid)
    x_grid, y_grid, z_grid = x_grid.to(device), y_grid.to(device), z_grid.to(device)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid = z_grid.repeat(x.shape[0], 1, 1)
    stiffness = stiffness.repeat(x.shape[0], 1, 1)
    friction = friction.repeat(x.shape[0], 1, 1)
    H, W = int(2 * dphys_cfg.d_max / dphys_cfg.grid_res), int(2 * dphys_cfg.d_max / dphys_cfg.grid_res)
    assert x_grid.shape == (B, H, W)
    assert y_grid.shape == (B, H, W)
    assert z_grid.shape == (B, H, W)
    assert stiffness.shape == (B, H, W)
    assert friction.shape == (B, H, W)

    # simulate the rigid body dynamics
    with torch.no_grad():
        dphysics = DPhysics(dphys_cfg, device=device)
        states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0,
                                  stiffness=stiffness, friction=friction)

    # visualize using matplotlib
    for b in range(len(states[0])):
        # get the states and forces for the b-th rigid body and move them to the cpu
        xs, R, xds, omegas, x_points = [s[b].cpu().numpy() for s in states]
        F_spring, F_friction = [f[b].cpu().numpy() for f in forces]
        x_grid_np, y_grid_np, z_grid_np = [g[b].cpu().numpy() for g in [x_grid, y_grid, z_grid]]

        # visualize trajectory
        visualize_traj(states=(xs, R, xds, omegas, x_points),
                       x_grid=x_grid_np, y_grid=y_grid_np, z_grid=z_grid_np,
                       forces=(F_spring, F_friction),
                       vis_step=10)


def shoot_multiple():
    from time import time
    from scipy.spatial.transform import Rotation
    from monoforce.models.dphysics import vw_to_track_vel
    from monoforce.vis import set_axes_equal
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')

    # simulation parameters
    dphys_cfg = DPhysConfig()
    device = torch.device('cpu')
    dt = dphys_cfg.dt
    T = dphys_cfg.traj_sim_time
    num_trajs = dphys_cfg.n_sim_trajs
    vel_max, omega_max = dphys_cfg.vel_max, dphys_cfg.omega_max

    # rigid body parameters
    x_points = torch.as_tensor(dphys_cfg.robot_points, device=device)

    # initial state
    x = torch.tensor([[0.0, 0.0, 0.2]], device=device).repeat(num_trajs, 1)
    xd = torch.zeros_like(x)
    R = torch.eye(3, device=device).repeat(x.shape[0], 1, 1)
    # R = torch.tensor(Rotation.from_euler('z', np.pi/6).as_matrix(), dtype=torch.float32, device=device).repeat(num_trajs, 1, 1)
    omega = torch.zeros_like(x)
    x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)

    # terrain properties
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2).to(device)

    stiffness = dphys_cfg.k_stiffness * torch.ones_like(z_grid)
    friction = dphys_cfg.k_friction * torch.ones_like(z_grid)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid = z_grid.repeat(x.shape[0], 1, 1)
    stiffness = stiffness.repeat(x.shape[0], 1, 1)
    friction = friction.repeat(x.shape[0], 1, 1)

    # control inputs in m/s and rad/s
    assert num_trajs % 2 == 0, 'num_trajs must be even'
    vels_x = torch.cat([-vel_max * torch.ones((num_trajs // 2, int(T / dt))),
                        vel_max * torch.ones((num_trajs // 2, int(T / dt)))])
    omegas_z = torch.cat([torch.linspace(-omega_max, omega_max, num_trajs // 2),
                          torch.linspace(-omega_max, omega_max, num_trajs // 2)])
    assert vels_x.shape == (num_trajs, int(T / dt))
    assert omegas_z.shape == (num_trajs,)
    vels = torch.zeros((num_trajs, int(T / dt), 3))
    vels[:, :, 0] = vels_x
    omegas = torch.zeros((num_trajs, 3))
    omegas[:, 2] = omegas_z

    controls = torch.zeros((num_trajs, int(T / dt), 2))
    for i in range(num_trajs):
        controls[i, :, 0], controls[i, :, 1] = vw_to_track_vel(vels[i, :, 0], omegas[i, 2])
    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # put tensors to device
    state0 = tuple([s.to(device) for s in state0])
    z_grid = z_grid.to(device)
    controls = controls.to(device)

    # create the dphysics model
    dphysics = DPhysics(dphys_cfg, device=device)

    # simulate the rigid body dynamics
    with torch.no_grad():
        t0 = time()
        states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0)
        t1 = time()
        Xs, Xds, Rs, Omegas, X_points = states
        print(Xs.shape)
        print(f'Simulation took {(t1-t0):.3f} [sec] on device: {device}')

    # visualize
    with torch.no_grad():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # plot heightmap
        ax.plot_surface(x_grid[0].cpu().numpy(), y_grid[0].cpu().numpy(), z_grid[0].cpu().numpy(), alpha=0.6, cmap='terrain')
        set_axes_equal(ax)
        for i in range(num_trajs):
            ax.plot(Xs[i, :, 0].cpu(), Xs[i, :, 1].cpu(), Xs[i, :, 2].cpu(), c='b')
        ax.set_title(f'Simulation of {num_trajs} trajs (T={T} [sec] long) took {(t1-t0):.3f} [sec] on device: {device}')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.show()


if __name__ == '__main__':
    motion()
    # shoot_multiple()
