import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def heatmap(distribution_func, axis3d, **params):
    count = 200
    theta, phi = np.linspace(0, 2 * np.pi, count), np.linspace(0, np.pi, count)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # data values
    x_grid = np.zeros((count, count))
    y_grid = np.zeros((count, count))
    z_grid = np.zeros((count, count))

    data = np.zeros((count, count))
    for j, theta_v in enumerate(zip(sin_theta, cos_theta)):
        for i, phi_v in enumerate(zip(sin_phi, cos_phi)):
            x = phi_v[0] * theta_v[1]
            y = phi_v[0] * theta_v[0]
            z = phi_v[1]

            x_grid[i, j] = x
            y_grid[i, j] = y
            z_grid[i, j] = z

            data[i, j] = distribution_func([x, y, z], **params)

    # face colors
    coolwarm = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
    fcolors = coolwarm(norm(data))

    # plot
    axis3d.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, cmap=coolwarm, facecolors=fcolors,
                        linewidth=0, antialiased=False)
