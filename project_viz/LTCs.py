import glm
import matplotlib.pyplot as plt
import numpy as np

from umath import spherical_plot


def do_clamped_cosine(w):
    return np.maximum(0, w[2] / np.pi)


class LTC:
    def __init__(self, mat, d_origin):
        self.mat = mat
        self.do = d_origin
        self.det = glm.determinant(mat)

    def evaluate(self, w):
        w_p = self.mat * w
        length = glm.length(w_p)

        Jacobian = self.det / (length ** 3)
        return self.do(w_p / length) / Jacobian


if __name__ == '__main__':

    # subplot: cosine
    fig = plt.figure(figsize=[10, 10 * 2])
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    spherical_plot.heatmap(do_clamped_cosine, ax1)
    plt.title('Clamped Cosine Distribution')

    # subplot LTC
    mat = glm.rotate(-np.pi / 4, glm.vec3(0, 1, 0)) * glm.scale(glm.vec3(0.5, 1, 1))
    ltc = LTC(mat, do_clamped_cosine)

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    spherical_plot.heatmap(lambda w: ltc.evaluate(w), ax2)
    plt.title('LTC')

    plt.show()

