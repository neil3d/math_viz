import os
import site_config
import numpy as np
from umath import spherical_plot


def spherical_gaussian(direction, axis, sharpness, amplitude):
    return amplitude * np.exp(sharpness * (np.dot(axis, direction) - 1))


if __name__ == '__main__':
    fig, ax = spherical_plot.heatmap(spherical_gaussian, 'Spherical Gaussian',
                                     axis=[0, 0, 1], sharpness=3, amplitude=5)
    image_path = os.path.join(site_config.plot_output_path, 'spherical_gaussian.png')
    fig.savefig(image_path)
    print(image_path, 'saved.')
