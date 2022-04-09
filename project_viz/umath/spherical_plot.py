import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def heatmap(distribution_func, axis3d, **params):
    count = 200
    theta, phi = np.linspace(0, 2 * np.pi, count), np.linspace(0, np.pi, count)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # data values
    data = np.zeros((count, count))
    for j, theta_v in enumerate(zip(sin_theta, cos_theta)):
        for i, phi_v in enumerate(zip(sin_phi, cos_phi)):
            x = phi_v[0] * theta_v[1]
            y = phi_v[0] * theta_v[0]
            z = phi_v[1]
            data[i][j] = distribution_func([x, y, z], **params)

    # mesh grid
    sin_theta_grid, sin_phi_grid = np.meshgrid(sin_theta, sin_phi)
    cos_theta_grid, cos_phi_grid = np.meshgrid(cos_theta, cos_phi)

    x_grid = sin_phi_grid * cos_theta_grid
    y_grid = sin_phi_grid * sin_theta_grid
    z_grid = cos_phi_grid

    # face colors
    coolwarm = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
    fcolors = coolwarm(norm(data))

    # plot
    axis3d.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, cmap=coolwarm, facecolors=fcolors,
                    linewidth=0, antialiased=False)

