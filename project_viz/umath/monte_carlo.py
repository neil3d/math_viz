import numpy as np


def integrate(distribution, rand_size, num_samples, **params):
    integral = 0.0

    for _ in range(0, num_samples):
        u = np.random.uniform(0.0, 1.0, rand_size)
        sample = distribution.sample(u)
        val = distribution.evaluate(sample)
        pdf = distribution.pdf(sample, **params)
        integral += val / pdf

    return integral / num_samples


if __name__ == '__main__':
    import scipy.integrate

    # unit test: uniform sampling for testing, integral should be 1
    # see: https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables
    class PowerDistribution:
        def __init__(self, n, a, b):
            self.n = n
            self.a = a
            self.b = b
            n1 = n + 1
            self.c = n1 / (b ** n1 - a ** n1)

        def sample(self, u):
            return u * (self.b - self.a) + self.a

        def evaluate(self, x):
            n = self.n
            c = self.c
            return c * (x ** n)

        def pdf(self, x):
            return 1 / (self.b - self.a)


    a = np.random.randint(0, 10)
    b = np.random.randint(11, 20)
    n = np.random.randint(2, 10)
    print('$\\int_{a}^{b} x^{n} dx$'.format(a=a, b=b, n=n))

    d_power = PowerDistribution(n, a, b)

    ans_quad = scipy.integrate.quad(lambda x: d_power.evaluate(x), a, b)
    print('scipy.integrate.quad = ', ans_quad)

    ans_mc = integrate(d_power, None, 50000)
    print('Monte Carlo = ', (ans_mc, 1 - ans_mc))
