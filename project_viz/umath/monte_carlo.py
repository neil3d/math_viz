import numpy as np


def integrate(distribution, rand_size, num_samples):
    integral = 0.0

    for _ in range(0, num_samples):
        u = np.random.uniform(0, 1, rand_size)
        sample = distribution.sample(u)
        val = distribution.evaluate(sample)
        pdf = distribution.pdf(sample)
        integral += val / pdf

    return integral / num_samples
