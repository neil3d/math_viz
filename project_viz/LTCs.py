import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import site_config
from umath import LTC, spherical_plot


def do_clamped_cosine(w):
    return np.maximum(0, w[2] / np.pi)


def anim_view_rotation(fig, ax):
    def anim_tick(i):
        ax.view_init(45, i)

    anim = animation.FuncAnimation(fig, anim_tick, frames=range(30, 360 + 30, 10))
    anim.save('LTC_view.mp4', fps=30,
              progress_callback=lambda i, n: print('rendering {0} of {1}'.format(i, n))
              )
    print('done')


if __name__ == '__main__':
    # subplot: cosine
    fig = plt.figure(figsize=[10, 10 * 2])
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    spherical_plot.heatmap(do_clamped_cosine, ax1)
    plt.title('Clamped Cosine Distribution')

    # subplot LTC
    amplitude = 1
    forward = np.array([0, 1, 0], dtype='float')
    up = np.array([0, 0, 1], dtype='float')
    scale = np.array([0.75, 1, 1], dtype='float')
    skew = 0.5
    ltc = LTC.LTC(amplitude, forward, up, scale, skew)

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')

    spherical_plot.heatmap(lambda w: ltc.evaluate(w), ax2)
    plt.title('Linearly Transformed Cosines')
    plt.show()

    # save figure as image
    image_path = os.path.join(site_config.plot_output_path, 'LTCs.png')
    fig.savefig(image_path)
    print(image_path, 'saved.')

    # animation
    # anim_view_rotation(fig, ax2)
