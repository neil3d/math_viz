import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.spatial.transform import Rotation as R

from umath import LTC, spherical_plot


def do_clamped_cosine(w):
    return np.maximum(0, w[2] / np.pi)


def save_figure(output_path):
    # plot cosine
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    spherical_plot.heatmap(do_clamped_cosine, ax)
    plt.title('Clamped Cosine Distribution')

    image_path = os.path.join(output_path, 'clamped_cosine.png')
    fig.savefig(image_path)
    print('Figure saved: ', image_path)

    # plot LTC
    fig.clear()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    amplitude = 1
    forward = np.array([0, 1, 0], dtype='float')
    up = np.array([0, 0, 1], dtype='float')
    scale = np.array([1, 1, 1], dtype='float')
    skew = 0
    ltc = LTC.LTC(amplitude, forward, up, scale, skew)

    plt.title('Linearly Transformed Cosines')
    spherical_plot.heatmap(lambda w: ltc.evaluate(w), ax)

    image_path = os.path.join(output_path, 'LTC.png')
    fig.savefig(image_path)
    print('Figure saved: ', image_path)


def save_animation(output_path):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    def anim_ltc(i):
        pitch = i * 10
        yaw = 0
        roll = 0

        mat_rot = R.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()

        amplitude = 1
        forward = mat_rot[1]
        up = mat_rot[2]

        s = 0.2+(np.sin(np.deg2rad(pitch))+1)*0.5
        scale = np.array([s, 1, 1], dtype='float')
        skew = 0
        ltc = LTC.LTC(amplitude, forward, up, scale, skew)

        ax.clear()
        plt.title('Linearly Transformed Cosines, Rotation=({pitch},{yaw},{roll}), scale[x]={scale:.2f}'.format(
            pitch=pitch, yaw=yaw, roll=roll, scale=s
        ))
        spherical_plot.heatmap(lambda w: ltc.evaluate(w), ax)

    star_time = time.perf_counter()
    anim = animation.FuncAnimation(fig, anim_ltc, frames=range(0, 36))

    anim_path = os.path.join(output_path, 'LTC.mp4')
    print('Rendering {0} ...'.format(anim_path))
    anim.save(anim_path, fps=24,
              progress_callback=lambda i, n: print('\t Frame {0} of {1} ...'.format(i, n), end='\r')
              )
    time_cost = time.perf_counter() - star_time
    print('\n\t Done, {0:.2f} seconds'.format(time_cost))
    print('Animation saved: ', anim_path)


if __name__ == '__main__':
    save_figure('./')
    save_animation('./')
