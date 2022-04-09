import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from umath import spherical_plot


def do_clamped_cosine(w):
    return np.maximum(0, w[2] / np.pi)


def normalize(vec):
    return vec / np.linalg.norm(vec)


class LTC:
    def __init__(self, amplitude, forward, up, scale, skew):
        self.amplitude = amplitude
        self.forward = normalize(forward)
        self.up = normalize(up)
        self.right = np.cross(self.forward, self.up)
        self.up = np.cross(self.right, self.forward)

        mat_rot = np.matrix([self.right, self.forward, self.up])
        mat_scale_skew = np.matrix([
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [skew, 0, scale[2]]
        ])

        self.mat = np.matmul(mat_rot, mat_scale_skew)
        self.mat_inv = np.linalg.inv(self.mat)
        self.det = np.abs(np.linalg.det(self.mat))

    def evaluate(self, w):
        return w[2]


def anim_view_rotation(fig, ax):
    def anim_tick(i):
        ax.view_init(45, i)

    anim = animation.FuncAnimation(fig, anim_tick, frames=range(30, 360+30, 5))
    anim.save('LTC_view.mp4', fps=30,
              progress_callback=lambda i, n: print('rendering {0} of {1}'.format(i, n))
              )
    print('done')


if __name__ == '__main__':
    # subplot: cosine
    fig = plt.figure(figsize=[10, 10])
    # ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    # spherical_plot.heatmap(do_clamped_cosine, ax1)
    # plt.title('Clamped Cosine Distribution')

    # subplot LTC
    amplitude = 1
    forward = np.array([0, 1, 0], dtype='float')
    up = np.array([0, 0, 1], dtype='float')
    scale = np.array([1, 1, 1], dtype='float')
    skew = 0.0
    ltc = LTC(amplitude, forward, up, scale, skew)

    ax2 = fig.add_subplot(1, 1, 1, projection='3d')

    spherical_plot.heatmap(lambda w: ltc.evaluate(w), ax2)
    plt.title('Linearly Transformed Cosines')
    plt.show()

    # animation
    anim_view_rotation(fig, ax2)
