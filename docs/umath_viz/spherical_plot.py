import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm

def heatmap(distribution_func, **params):
    COUNT = 200
    Theta, Phi = np.linspace(0, 2 * np.pi, COUNT), np.linspace(0, np.pi, COUNT)
    dist = np.zeros((COUNT,COUNT))
    
    # values
    for j, theta in enumerate(Theta):
        for i,phi in enumerate(Phi):
            x = np.sin(phi)*np.cos(theta)
            y = np.sin(phi)*np.sin(theta)
            z = np.cos(phi)
            dist[i][j] = distribution_func([x,y,z], **params)

    # mesh grid
    THETA, PHI = np.meshgrid(Theta, Phi)
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)
    
    # face colors
    coolwarm = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=dist.min(), vmax=dist.max())
    fcolors = coolwarm(norm(dist))

    # plot
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=coolwarm, facecolors=fcolors,
        linewidth=0, antialiased=False)

    plt.title(distribution_func.__name__)
    plt.show()

