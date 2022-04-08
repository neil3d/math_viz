import os
import site_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from umath import spherical_plot
import glm


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
    # spherical_plot.heatmap(do_clamped_cosine, 'Clamped Cosine Distribution')

    mat = glm.rotate(-np.pi / 3, glm.vec3(0, 1, 0)) * glm.scale(glm.vec3(0.5, 1, 1))
    ltc = LTC(mat, do_clamped_cosine)
    spherical_plot.heatmap(lambda w: ltc.evaluate(w), 'LTC')
