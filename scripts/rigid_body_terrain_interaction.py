import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set matplotlib backend to show plots in a separate window
plt.switch_backend('Qt5Agg')


g = 9.81
dt = 0.01
T = 10.0
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

def rigid_body_motion():
    # sample x_points from a sphere
    n_particles = 36
    theta = torch.linspace(0, 2 * np.pi, int(np.sqrt(n_particles)))
    phi = torch.linspace(0, np.pi, int(np.sqrt(n_particles)))
    theta, phi = torch.meshgrid(theta, phi)
    r = 0.5
    X = r * torch.sin(phi) * torch.cos(theta)
    Y = r * torch.sin(phi) * torch.sin(theta)
    Z = r * torch.cos(phi)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    x_points = torch.stack([X, Y, Z], dim=-1)

    # # vertices of a cube
    # x_points = torch.tensor([
    #     [1.0, 1.0, 1.0],
    #     [1.0, 1.0, -1.0],
    #     [1.0, -1.0, 1.0],
    #     [1.0, -1.0, -1.0],
    #     [-1.0, 1.0, 1.0],
    #     [-1.0, 1.0, -1.0],
    #     [-1.0, -1.0, 1.0],
    #     [-1.0, -1.0, -1.0],
    # ])
    # # transform cube to a parallelepiped
    # x_points = x_points * torch.tensor([2.0, 1.0, 0.5])

    print('Points shape:', x_points.shape)

    # initial position and velocity
    x0 = torch.tensor([0.1, 0.0, 5.0])
    v0 = torch.tensor([0.0, 0.0, 0.0])
    omega0 = torch.tensor([0.0, 0.0, 0.0])
    x_points = x_points + x0

    # center of gravity
    x = x_points.mean(dim=0)
    print('Center of gravity:', x)

    # heightmap defining the terrain
    x_grid = torch.arange(-d_max, d_max, grid_res)
    y_grid = torch.arange(-d_max, d_max, grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # z_grid = torch.zeros(x_grid.shape)
    # z_grid = torch.sin(x_grid) + torch.cos(y_grid)
    z_grid = torch.exp(-x_grid ** 2 / 10) * torch.exp(-y_grid ** 2 / 10)

    # plot results
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # motion of the rigid body affected by gravity
    xd = v0
    omega = omega0
    xs = []
    m = 1.0
    I = 200. * torch.eye(3)  # inertia tensor
    I_inv = torch.inverse(I)
    k_stiffness = 100.0
    k_damping = 5.0
    for i in range(int(T / dt)):
        # check if the rigid body is in contact with the terrain
        z = interpolate_height(x_grid, y_grid, z_grid, x_points[:, 0], x_points[:, 1])
        dh = x_points[:, 2:3] - z[:, None]
        # in_contact = torch.sigmoid(-dh)
        in_contact = (dh <= 0).float()

        # normals of the terrain at the contact points
        dz_dx = interpolate_height(x_grid, y_grid, z_grid, x_points[:, 0] + grid_res, x_points[:, 1]) - z[:]
        dz_dy = interpolate_height(x_grid, y_grid, z_grid, x_points[:, 0], x_points[:, 1] + grid_res) - z[:]
        t1 = torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx), dz_dx])
        t2 = torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy), dz_dy])
        n = torch.cross(t1, t2).T
        assert n.shape == x_points.shape
        n = n / torch.norm(n, dim=1, keepdim=True)
        n = n * in_contact

        # reaction at the contact points as spring-damper forces
        F_spring = -(k_stiffness * dh + k_damping * xd) * n * in_contact
        # reaction forces always point upwards
        z_axis = torch.tensor([0.0, 0.0, 1.0]).view(3, 1)
        F_spring = F_spring * torch.sign(F_spring @ z_axis)
        # # limit too large forces beetwen 2 and 98 quantile
        F_spring = torch.clamp(F_spring, torch.quantile(F_spring, 0.02), torch.quantile(F_spring, 0.98))

        # rigid body rotation
        torque = torch.sum(torch.cross(x_points - x, F_spring), dim=0)
        omega_d = I_inv @ torque
        omega = integration_step(omega, omega_d, dt)

        # motion of the cog
        F_grav = torch.tensor([0.0, 0.0, -m * g])
        F_cog = F_grav + F_spring.mean(dim=0)
        xdd = F_cog / m
        xd = integration_step(xd, xdd, dt)

        # motion of point composed of cog motion and rotation of the rigid body
        xd_points = xd + torch.cross(omega[None], x_points - x)  # Koenig's theorem in mechanics
        x_points = integration_step(x_points, xd_points, dt)
        x = x_points.mean(dim=0)

        xs.append(x)

        if i % 10 == 0:
            # plot particles
            plt.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # set axis limits
            ax.set_xlim(-d_max, d_max)
            ax.set_ylim(-d_max, d_max)
            ax.set_zlim(-d_max, d_max)
            ax.scatter(x_points[:, 0].numpy(), x_points[:, 1].numpy(), x_points[:, 2].numpy())
            ax.scatter(x[0].item(), x[1].item(), x[2].item(), c='r')
            # plot trajectory
            xs_tensor = torch.stack(xs)
            ax.plot(xs_tensor[:, 0].detach().numpy(), xs_tensor[:, 1].detach().numpy(), xs_tensor[:, 2].detach().numpy(), c='b')
            # plot terrain
            ax.plot_surface(x_grid.numpy(), y_grid.numpy(), z_grid.numpy(), alpha=0.5, cmap='viridis')

            # plot normals
            ax.quiver(x_points[:, 0], x_points[:, 1], x_points[:, 2], F_spring[:, 0], F_spring[:, 1], F_spring[:, 2], color='r')

            set_axes_equal(ax)
            plt.pause(0.01)

    plt.show()

def main():
    # forward()
    # learn_height()
    # learn_terrain_properties()
    rigid_body_motion()


if __name__ == '__main__':
    main()
