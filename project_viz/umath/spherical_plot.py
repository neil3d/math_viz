import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def heatmap(distribution_func, title, **params):
    count = 200
    theta, phi = np.linspace(0, 2 * np.pi, count), np.linspace(0, np.pi, count)
    dist = np.zeros((count, count))

    # values
    for j, th in enumerate(theta):
        for i, ph in enumerate(phi):
            x = np.sin(ph) * np.cos(th)
            y = np.sin(ph) * np.sin(th)
            z = np.cos(ph)
            dist[i][j] = distribution_func([x, y, z], **params)

    # mesh grid
    THETA, PHI = np.meshgrid(theta, phi)
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)

    # face colors
    coolwarm = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=dist.min(), vmax=dist.max())
    fcolors = coolwarm(norm(dist))

    # plot
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=coolwarm, facecolors=fcolors,
                    linewidth=0, antialiased=False)

    plt.title(title)
    plt.show()

    return fig, ax
