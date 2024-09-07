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

def interpolate_height(x_grid, y_grid, z_grid, x_query, y_query):
    """
    Interpolates the height at the desired (x_query, y_query) coordinates.

    Parameters:
    - x_grid: Tensor of x coordinates (2D array).
    - y_grid: Tensor of y coordinates (2D array).
    - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (2D array).
    - x_query: Tensor of desired x coordinates for interpolation (1D or 2D array).
    - y_query: Tensor of desired y coordinates for interpolation (1D or 2D array).

    Returns:
    - Interpolated z values at the queried coordinates.
    """

    # Ensure inputs are tensors
    x_grid = torch.as_tensor(x_grid)
    y_grid = torch.as_tensor(y_grid)
    z_grid = torch.as_tensor(z_grid)
    x_query = torch.as_tensor(x_query)
    y_query = torch.as_tensor(y_query)

    # Normalize query coordinates to [-1, 1] range
    x_min, x_max = x_grid.min(), x_grid.max()
    y_min, y_max = y_grid.min(), y_grid.max()

    x_query_normalized = 2.0 * (x_query - x_min) / (x_max - x_min) - 1.0
    y_query_normalized = 2.0 * (y_query - y_min) / (y_max - y_min) - 1.0

    # Create normalized query grid
    query_grid = torch.stack([x_query_normalized, y_query_normalized], dim=-1).unsqueeze(0).unsqueeze(0)

    # Add batch and channel dimensions to z_grid
    z_grid = z_grid.unsqueeze(0).unsqueeze(0)

    # Perform grid sampling (bilinear interpolation)
    interpolated_z = torch.nn.functional.grid_sample(z_grid, query_grid, mode='bilinear', align_corners=True)

    # Remove unnecessary dimensions
    interpolated_z = interpolated_z.squeeze()

    return interpolated_z


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


def surface_normals(x_grid, y_grid, z_grid, x_query, y_query):
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - x_grid: Tensor of x coordinates (2D array).
    - y_grid: Tensor of y coordinates (2D array).
    - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (2D array).
    - x_query: Tensor of desired x coordinates for interpolation (1D or 2D array).
    - y_query: Tensor of desired y coordinates for interpolation (1D or 2D array).

    Returns:
    - Surface normals and tangents at the queried coordinates.
    """
    # Interpolate the height at the queried coordinates
    x_i = (x_query + d_max) / grid_res
    y_i = (y_query + d_max) / grid_res
    x_i = torch.clamp(x_i, 0, len(x_grid) - 2).long()
    y_i = torch.clamp(y_i, 0, len(y_grid) - 2).long()
    dz_dx = (z_grid[x_i + 1, y_i] - z_grid[x_i , y_i]) / grid_res
    dz_dy = (z_grid[x_i, y_i + 1] - z_grid[x_i, y_i]) / grid_res

    # n = [-dz_dx, -dz_dy, 1]
    n = torch.stack([
        -dz_dx,
        -dz_dy,
        torch.ones_like(x_i)
    ], dim=-1)
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
    # x_points = torch.tensor(x_points, dtype=torch.float32)

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
    mask_left = x_points[:, 1] > cog[1]
    mask_right = x_points[:, 1] < cog[1]

    m = 40.0
    I = inertia_tensor(m, x_points)
    I *= 100.0  # scale the inertia tensor as the point cloud is sparse

    return x_points, m, I, mask_left, mask_right


def heightmap(d_max, grid_res):
    x_grid = torch.arange(-d_max, d_max, grid_res)
    y_grid = torch.arange(-d_max, d_max, grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # z_grid = torch.zeros(x_grid.shape)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    z_grid = torch.exp(-(x_grid-2) ** 2 / 2) * torch.exp(-y_grid ** 2 / 4)

    return x_grid, y_grid, z_grid


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector.

    Parameters:
    - v: Input vector.

    Returns:
    - Skew-symmetric matrix of the input vector.
    """
    assert v.shape == (3,)
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def forward_kinematics(x, xd, R, omega, x_points, xd_points,
                       x_grid, y_grid, z_grid,
                       m, I_inv, mask_left, mask_right, u_left, u_right,
                       k_stiffness, k_damping, k_friction):
    # check if the rigid body is in contact with the terrain
    z_points = interpolate_height(x_grid, y_grid, z_grid, x_points[:, 0], x_points[:, 1])
    dh_points = x_points[:, 2:3] - z_points[:, None]
    # in_contact = torch.sigmoid(-dh_points)
    in_contact = (dh_points <= 0.0).float()

    # compute surface normals at the contact points
    n = surface_normals(x_grid, y_grid, z_grid, x_points[:, 0], x_points[:, 1])

    # reaction at the contact points as spring-damper forces
    xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
    F_spring = -(k_stiffness * dh_points + k_damping * xd_points_n) * n * in_contact  # F_s = -k * dh - b * v_n
    # limit the spring forces
    F_spring = torch.clamp(F_spring, min=0.0, max=2 * m * g)

    # friction forces: https://en.wikipedia.org/wiki/Friction
    N = torch.norm(F_spring, dim=-1, keepdim=True)
    xd_points_tau = xd_points - xd_points_n * n  # tangential velocities at the contact points
    tau = normailized(xd_points_tau)  # tangential directions of the velocities
    F_friction = -k_friction * N * tau  # F_fr = -k_fr * N * tau

    # thrust forces: left and right
    thrust_dir = normailized(R @ torch.tensor([1.0, 0.0, 0.0]))
    x_left = x_points[mask_left].mean(dim=0)  # left thrust is applied at the mean of the left points
    x_right = x_points[mask_right].mean(dim=0)  # right thrust is applied at the mean of the right points
    F_thrust_left = u_left * thrust_dir * in_contact[mask_left].any()  # F_l = u_l * thrust_dir
    F_thrust_right = u_right * thrust_dir * in_contact[mask_right].any()  # F_r = u_r * thrust_dir
    torque_left = torch.cross(x_left - x, F_thrust_left)  # M_l = (x_l - x) x F_l
    torque_right = torch.cross(x_right - x, F_thrust_right)  # M_r = (x_r - x) x F_r
    torque_thrust = torque_left + torque_right  # M_thrust = M_l + M_r

    # rigid body rotation
    torque = torch.sum(torch.cross(x_points - x, F_spring + F_friction), dim=0) + torque_thrust  # M = sum(r_i x F_i)
    omega_d = I_inv @ torque  # omega_d = I^(-1) M
    omega_skew = skew_symmetric(omega)  # omega_skew = [omega]_x
    dR = omega_skew @ R  # dR = [omega]_x R

    # motion of the cog
    F_grav = torch.tensor([0.0, 0.0, -m * g])
    F_cog = F_grav + F_spring.sum(dim=0) + F_friction.sum(dim=0) + F_thrust_left + F_thrust_right  # ma = sum(F_i)
    xdd = F_cog / m  # a = F / m

    # motion of point composed of cog motion and rotation of the rigid body
    xd_points = xd + torch.cross(omega.view(1, 3), x_points - x)  # Koenig's theorem in mechanics

    return xd, xdd, dR, omega_d, xd_points, F_spring, F_friction, F_thrust_left, F_thrust_right


def update_states(x, xd, xdd, R, dR, omega, omega_d, x_points, xd_points, dt):
    xd = integration_step(xd, xdd, dt)
    x = integration_step(x, xd, dt)
    x_points = integration_step(x_points, xd_points, dt)
    omega = integration_step(omega, omega_d, dt)
    R = integration_step(R, dR, dt)

    return x, xd, R, omega, x_points

def dphysics(state, xd_points,
             x_grid, y_grid, z_grid,
             m, I, mask_left, mask_right, controls,
             k_stiffness=100., k_damping=None, k_friction=0.5,
             T=10.0, dt=0.01):
    # state: x, xd, R, omega, x_points
    x, xd, R, omega, x_points = state

    I_inv = torch.inverse(I)
    if k_damping is None:
        k_damping = np.sqrt(4 * m * k_stiffness)  # critically damping

    # dynamics of the rigid body
    states = []
    forces = []
    for i in range(int(T / dt)):
        # control inputs
        u_left, u_right = controls[i]  # thrust forces, Newtons or kg*m/s^2
        # forward kinematics
        (xd, xdd, dR, omega_d, xd_points,
         F_spring, F_friction, F_thrust_left, F_thrust_right) = forward_kinematics(x, xd, R, omega, x_points, xd_points,
                                                                                   x_grid, y_grid, z_grid,
                                                                                   m, I_inv, mask_left, mask_right, u_left,
                                                                                   u_right,
                                                                                   k_stiffness, k_damping, k_friction)
        # update states: integration steps
        x, xd, R, omega, x_points = update_states(x, xd, xdd, R, dR, omega, omega_d, x_points, xd_points, dt)

        # store states
        states.append((x, xd, R, omega, x_points))

        # store forces
        forces.append((F_spring, F_friction, F_thrust_left, F_thrust_right))

    return states, forces


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
    T = 10.0

    # control inputs
    controls = 2.6 * torch.tensor([[10.0, 10.0]] * int(T / dt))

    # rigid body parameters
    x_points, m, I, mask_left, mask_right = rigid_body_params()

    # terrain parameters
    k_stiffness = 100.0
    k_damping = np.sqrt(4 * m * k_stiffness)
    k_friction = 0.51

    # initial state
    x = torch.tensor([-2.0, 0.0, 1.0])
    xd = torch.tensor([0.0, 0.0, 0.0])
    R = torch.eye(3)
    omega = torch.tensor([0.0, 0.0, 0.0])
    x_points = x_points @ R.T + x
    xd_points = xd.repeat(len(x_points), 1)

    # heightmap defining the terrain
    x_grid, y_grid, z_grid = heightmap(d_max, grid_res)

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # simulate the rigid body dynamics
    states, forces = dphysics(state0, xd_points,
                              x_grid, y_grid, z_grid,
                              m, I, mask_left, mask_right,
                              controls,
                              k_stiffness=k_stiffness, k_damping=k_damping, k_friction=k_friction,
                              T=T, dt=dt)
    # visualize
    visualize(states, x_grid, y_grid, z_grid, forces=forces, vis_step=10, mask_left=mask_left, mask_right=mask_right)
    

def visualize(states, x_grid, y_grid, z_grid, forces=None, vis_step=1, mask_left=None, mask_right=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        for i, state in enumerate(states):
            if i % vis_step != 0:
                continue
            x, xd, R, omega, x_points = state

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
            xs_tensor = torch.stack([s[0] for s in states])
            ax.plot(xs_tensor[:, 0].cpu(), xs_tensor[:, 1].cpu(), xs_tensor[:, 2].cpu(), c='b')

            # plot cog velocity
            ax.quiver(x[0], x[1], x[2], xd[0], xd[1], xd[2], color='k')

            # plot terrain
            ax.plot_surface(x_grid.cpu(), y_grid.cpu(), z_grid.cpu(), alpha=0.5, cmap='terrain')
            
            # plot forces
            if forces is not None:
                assert len(forces) == len(states)
                F = forces[i]
                F_spring, F_friction, F_thrust_left, F_thrust_right = F
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
    T = 10.0
    vis = True
    n_iters = 100
    lr = 0.002

    # rigid body parameters
    x_points, m, I, mask_left, mask_right = rigid_body_params()

    # initial state
    x = torch.tensor([-1.0, 0.0, 1.0])
    xd = torch.tensor([0.0, 0.0, 0.0])
    R = torch.eye(3)
    omega = torch.tensor([0.0, 0.0, 0.0])
    x_points = x_points @ R.T + x
    xd_points = xd.repeat(len(x_points), 1)

    # heightmap defining the terrain
    x_grid, y_grid, z_grid_gt = heightmap(d_max, grid_res)

    # control inputs
    controls = 2.6 * torch.tensor([[10.0, 10.0]] * int(T / dt))

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # simulate the rigid body dynamics
    states_gt, forces_gt = dphysics(state0, xd_points,
                                    x_grid, y_grid, z_grid_gt,
                                    m, I, mask_left, mask_right,
                                    controls,
                                    T=T, dt=dt)
    if vis:
        visualize(states_gt, x_grid, y_grid, z_grid_gt, forces_gt, vis_step=10)

    # initial guess for the heightmap
    z_grid = torch.zeros_like(z_grid_gt, requires_grad=True)

    # optimization
    optimizer = torch.optim.Adam([z_grid], lr=lr)
    z_grid_best = z_grid.clone()
    loss_best = np.inf
    for i in range(n_iters):
        optimizer.zero_grad()
        # simulate the rigid body dynamics
        states, forces = dphysics(state0, xd_points,
                                  x_grid, y_grid, z_grid,
                                  m, I, mask_left, mask_right,
                                  controls,
                                  T=T, dt=dt)
        # unroll the states
        xs, xds, Rs, omegas, x_points = zip(*states)
        xs_gt, xds_gt, Rs_gt, omegas_gt, x_points_gt = zip(*states_gt)

        # compute the loss
        xs, xds, Rs, omegas, x_points = torch.stack(xs), torch.stack(xds), torch.stack(Rs), torch.stack(omegas), torch.stack(x_points)
        xs_gt, xds_gt, Rs_gt, omegas_gt, x_points_gt = torch.stack(xs_gt), torch.stack(xds_gt), torch.stack(Rs_gt), torch.stack(omegas_gt), torch.stack(x_points_gt)
        loss_x = torch.nn.functional.mse_loss(xs, xs_gt)
        loss_xd = torch.tensor(0.0)  # torch.nn.functional.mse_loss(xds, xds_gt)
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
            print(f'Heightmap difference: {z_diff.item():.3f}')

        if vis and (i == 0 or i == n_iters - 1):
            visualize(states, x_grid, y_grid, z_grid_best, vis_step=10)


def main():
    motion()
    # optimization()


if __name__ == '__main__':
    main()
