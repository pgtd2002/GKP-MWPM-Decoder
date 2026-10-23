import numpy as np


def sample_gaussian_noise(dim, sigma, rng=None):
    """
    Sample a Gaussian random displacement vector.

    Parameters
    ----------
    dim : int
        Dimension of the noise vector (e.g., 2n for n oscillators).
    sigma : float
        Standard deviation of Gaussian noise.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    xi : np.ndarray
        Gaussian noise vector of shape (dim,)
    """

    if rng is None:
        rng = np.random.default_rng()

    xi = rng.normal(
        loc=0.0,
        scale=sigma,
        size=dim
    )

    return np.array(xi)