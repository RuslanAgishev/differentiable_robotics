import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set matplotlib backend to show plots in a separate window
plt.switch_backend('Qt5Agg')


g = 9.81
dt = 0.1
T = 4.0
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

def force(x, xd, m, x_grid, y_grid, z_grid, k_stiffness=100.0, k_damping=5.0, k_friction=0.5):
    """
    Calculates the force acting on the point mass.
    """
    # gravity force
    F_grav = torch.tensor([0.0, 0.0, -m * g])

    # get height of terrain at current position
    x_i, y_i = int((x[0].item() + d_max) / grid_res), int((x[1].item() + d_max) / grid_res)
    x_i, y_i = max(0, min(x_i, z_grid.shape[0]-2)), max(0, min(y_i, z_grid.shape[1]-2))
    # z = z_grid[x_i, y_i]
    # use interpolation to get height of terrain at current position
    z = interpolate_height(x_grid, y_grid, z_grid, [x[0]], [x[1]])

    # if no contact with terrain, return zero force
    dh = x[2] - z
    # in_contact = dh <= 0
    in_contact = torch.sigmoid(-dh)

    # calculate normal vector of terrain at current position
    dz_dx = (z_grid[x_i + 1, y_i] - z_grid[x_i - 1, y_i]) / 2
    dz_dy = (z_grid[x_i, y_i + 1] - z_grid[x_i, y_i - 1]) / 2
    n = torch.cross(torch.tensor([1.0, 0.0, dz_dx]), torch.tensor([0.0, 1.0, dz_dy]))
    n = n / torch.norm(n)

    # calculate terrain reaction force: underdamped spring-damper system: https://beltoforion.de/en/harmonic_oscillator/
    F_spring = -(k_stiffness * dh + k_damping * xd) * n * in_contact

    # friction force
    F_friction = -k_friction * xd * in_contact

    # total force
    F = F_grav + F_spring + F_friction

    return F

def step(x, xd, xdd, dt):
    xd = xd + xdd * dt
    x = x + xd * dt
    return x, xd

def step_runge_kutta(x, xd, xdd, dt):
    k1 = xdd * dt
    k2 = (xdd + k1 / 2) * dt
    k3 = (xdd + k2 / 2) * dt
    k4 = (xdd + k3) * dt
    xd = xd + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    k1 = xd * dt
    k2 = (xd + k1 / 2) * dt
    k3 = (xd + k2 / 2) * dt
    k4 = (xd + k3) * dt
    x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, xd


def dphysics(x, xd, x_grid, y_grid, z_grid, m=1.0, k_stiffness=100.0, k_damping=5.0, k_friction=0.5):
    """
    Simulates the motion of a point mass under the influence of gravity and terrain
    """
    xs = []
    vs = []
    forces = []
    for i in range(int(T / dt)):
        F = force(x, xd, m, x_grid, y_grid, z_grid, k_stiffness, k_damping, k_friction)
        xdd = F / m
        # x, xd = step(x, xd, xdd, dt)
        x, xd = step_runge_kutta(x, xd, xdd, dt)

        xs.append(x)
        vs.append(xd)
        forces.append(F)

    xs = torch.stack(xs)
    vs = torch.stack(vs)
    forces = torch.stack(forces)

    return xs, vs, forces


def forward():
    """
    point mass affected by gravity
    """
    # initial position and velocity
    x0 = torch.tensor([-4.0, 0.0, 5.0])
    v0 = torch.tensor([1.0, 0.2, 0.0])

    # heightmap defining the terrain
    x_grid = torch.arange(-d_max, d_max, grid_res)
    y_grid = torch.arange(-d_max, d_max, grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # z_grid = torch.zeros(x_grid.shape)
    # z_grid = torch.sin(x_grid) + torch.cos(y_grid)
    z_grid = torch.exp(-x_grid**2 / 10) * torch.exp(-y_grid**2 / 10)

    # simulate point mass motion
    xs, vs, forces = dphysics(x0, v0, x_grid, y_grid, z_grid, m=1)

    # plot results
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(y_grid.numpy(), x_grid.numpy(), z_grid.numpy(), alpha=0.5, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    for i in range(int(T / dt)):
        x = xs[i]
        v = vs[i]
        F = forces[i]

        if i % 1 == 0:
            # plot point mass
            ax.scatter(x[0].item(), x[1].item(), x[2].item(), c='r')
            # # plot total force
            # ax.quiver(x[0].item(), x[1].item(), x[2].item(),
            #           F[0].item() / g, F[1].item() / g, F[2].item() / g, color='m')
            # # plot velocity
            # ax.quiver(x[0].item(), x[1].item(), x[2].item(), v[0].item(), v[1].item(), v[2].item(), color='g')
            plt.pause(0.01)

    plt.show()


def learn_height():
    """
    optimize point mass trajectory
    """
    # initial position and velocity
    x0 = torch.tensor([-4.0, 0.0, 5.0])
    v0 = torch.tensor([1.0, 0.2, 0.0])

    # heightmap defining the terrain
    x_grid = torch.arange(-d_max, d_max, grid_res)
    y_grid = torch.arange(-d_max, d_max, grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    z_grid_gt = torch.sin(x_grid) + torch.cos(y_grid)

    # simulate point mass motion
    xs_gt, vs_gt, forces_gt = dphysics(x0, v0, x_grid, y_grid, z_grid_gt)

    # optimize heightmap to fit the trajectory
    z_grid = torch.zeros(x_grid.shape, requires_grad=True)
    optimizer = torch.optim.Adam([z_grid], lr=0.01)
    n_iters = 1_000
    loss_history = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        xs, vs, forces = dphysics(x0, v0, x_grid, y_grid, z_grid)
        loss_x = torch.sum((xs - xs_gt) ** 2)
        # loss_v = torch.sum((vs - vs_gt) ** 2)
        # loss = loss_x + loss_v
        loss = loss_x
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if i % 10 == 0:
            # print('Loss X:', loss_x.item())
            # print('Loss V:', loss_v.item())
            print('Loss:', loss.item())

    # plot results
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')
    # ax.plot_surface(y_grid.numpy(), x_grid.numpy(), z_grid_gt.numpy(), alpha=0.5, cmap='viridis')
    ax.plot_surface(y_grid.numpy(), x_grid.numpy(), z_grid.detach().numpy(), alpha=0.5, cmap='viridis')
    # plot trajectories
    ax.plot(xs_gt[:, 0].detach().numpy(), xs_gt[:, 1].detach().numpy(), xs_gt[:, 2].detach().numpy(), c='r')
    ax.plot(xs[:, 0].detach().numpy(), xs[:, 1].detach().numpy(), xs[:, 2].detach().numpy(), c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    ax = fig.add_subplot(122)
    ax.plot(loss_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    plt.show()


def learn_terrain_properties():
    """
    optimize terrain properties to fit the trajectory
    """
    # initial position and velocity
    x0 = torch.tensor([-4.0, 0.0, 5.0])
    v0 = torch.tensor([1.0, 0.2, 0.0])

    # heightmap defining the terrain
    x_grid = torch.arange(-d_max, d_max, grid_res)
    y_grid = torch.arange(-d_max, d_max, grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    z_grid = torch.sin(x_grid) + torch.cos(y_grid)

    # simulate point mass motion
    xs_gt, vs_gt, forces_gt = dphysics(x0, v0, x_grid, y_grid, z_grid)

    # optimize terrain and robot properties to fit the trajectory
    k_stiffness = torch.tensor(150.0, requires_grad=True)  # GT: 100.0
    m = torch.tensor(1.0, requires_grad=False)  # GT: 1.0
    omega0 = torch.sqrt(k_stiffness / m)
    k_damping = torch.tensor(0.3 * omega0, requires_grad=True)  # GT: 0.5 * omega0
    k_friction = torch.tensor(0.4, requires_grad=True)  # GT: 0.5
    optimizer = torch.optim.Adam([
        {'params': k_stiffness, 'lr': 0.1},
        {'params': k_damping, 'lr': 0.01},
        {'params': k_friction, 'lr': 0.01},
        # {'params': m, 'lr': 0.1}
    ])
    n_iters = 1_000
    loss_history = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        xs, vs, forces = dphysics(x0, v0, x_grid, y_grid, z_grid, m, k_stiffness, k_damping, k_friction)
        loss = torch.sum((xs - xs_gt) ** 2) + torch.sum((vs - vs_gt) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if i % 10 == 0:
            print('Loss:', loss.item())
            print('k_stiffness:', k_stiffness.item())
            print('k_damping:', k_damping.item())
            print('k_friction:', k_friction.item())
            print('m:', m.item())

        if i == 0 or i == n_iters-1:
            # plot results
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(121, projection='3d')
            ax.plot_surface(y_grid.numpy(), x_grid.numpy(), z_grid.numpy(), alpha=0.5, cmap='viridis')
            # plot trajectories
            ax.plot(xs_gt[:, 0].detach().numpy(), xs_gt[:, 1].detach().numpy(), xs_gt[:, 2].detach().numpy(), c='r')
            ax.plot(xs[:, 0].detach().numpy(), xs[:, 1].detach().numpy(), xs[:, 2].detach().numpy(), c='b')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            set_axes_equal(ax)

            ax = fig.add_subplot(122)
            ax.plot(loss_history)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')

            plt.show()


def main():
    # forward()
    # learn_height()
    learn_terrain_properties()


if __name__ == '__main__':
    main()
