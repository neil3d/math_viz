import os
import site_config
import numpy as np
import matplotlib.pyplot as plt

from umath import spherical_plot


def spherical_gaussian(direction, axis, sharpness, amplitude):
    return amplitude * np.exp(sharpness * (np.dot(axis, direction) - 1))


if __name__ == '__main__':
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    spherical_plot.heatmap(spherical_gaussian, ax,
                                     axis=[0, 0, 1], sharpness=3, amplitude=5)
    plt.title('Spherical Gaussian')
    plt.show()

    image_path = os.path.join(site_config.plot_output_path, 'spherical_gaussian.png')
    fig.savefig(image_path)
    print(image_path, 'saved.')
