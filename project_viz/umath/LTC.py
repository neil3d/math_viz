import numpy as np


class LTC:
    def __init__(self, forward, up, scale, skew):
        self.forward = LTC.normalize(forward)
        self.up = LTC.normalize(up)
        self.right = np.cross(self.forward, self.up)
        self.up = np.cross(self.right, self.forward)

        mat_rot = np.array([self.right, self.forward, self.up])
        mat_scale_skew = np.array([
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [skew, 0, scale[2]]
        ])

        self.mat = np.matmul(mat_rot, mat_scale_skew)
        self.mat_inv = np.linalg.inv(self.mat)
        self.det = np.abs(np.linalg.det(self.mat))

    @staticmethod
    def normalize(vec):
        return vec / np.linalg.norm(vec)

    def evaluate(self, w):
        l_o = LTC.normalize(np.dot(self.mat_inv, w))
        l = np.dot(self.mat, l_o)

        n = np.linalg.norm(l)
        Jacobian = self.det / (n ** 3)

        do = np.maximum(0, l_o[2] / np.pi)
        return do / Jacobian

    def sample(self, u):
        theta = np.arccos(np.sqrt(u[0]))
        phi = 2.0 * np.pi * u[1]
        s = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        return LTC.normalize(np.dot(self.mat, s))

    def pdf(self, w):
        return self.evaluate(w)


if __name__ == '__main__':
    import monte_carlo

    # test normalization
    forward = np.array([0, 1, 1], dtype='float')
    up = np.array([0, 0, 1], dtype='float')
    scale = np.array([0.5, 1.5, 1], dtype='float')
    skew = 0.1
    ltc = LTC(forward, up, scale, skew)

    ans = monte_carlo.integrate(ltc, 2, 10)
    print('integral = ', ans)
