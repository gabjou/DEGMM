import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split, normal, choice
from jax import vmap

class DataGenerator:
        def __init__(self, K, M, mus, Sigmas, pi, key=PRNGKey(0)):
                """
                Initialize the DataGenerator.

                Parameters:
                - K: Number of Gaussian components.
                - M: Number of ensemble members.
                - mus: List of mean vectors for each component (shape: (K, D)).
                - Sigmas: List of covariance matrices for each component (shape: (K, D, D)).
                - pi: Mixing coefficients for each component (shape: (K,)).
                - key: JAX random key.
                """
                self.K = K
                self.M = M
                self.D = mus.shape[1]
                self.mus = mus
                self.Sigmas = Sigmas
                self.pi = jnp.array(pi)
                self.key = key

        def generate_sample(self, key,k):
                """
                Generate a single sample.

                Parameters:
                - key: JAX random key.

                Returns:
                - sample: Generated sample (shape: (D,)).
                """
                key, subkey = split(key)
                sample = normal(subkey, shape=(self.D,)) @ jnp.linalg.cholesky(self.Sigmas[k,:,:]).T + self.mus[k,:]
                return sample

        def generate_data(self, n):
                """
                Generate an ensemble X of n rows, M columns, and dimension D.

                Parameters:
                - n: Number of samples to generate.

                Returns:
                - X: Generated data (shape: (n, M, D)).
                """
                def generate_row(key):
                        mkeys = split(key, self.M)
                        k = choice(key, jnp.arange(self.K), p=self.pi)
                        karray = jnp.full(self.M, k)
                        samples = vmap(self.generate_sample)(mkeys,karray)
                        return samples,k

                nkeys = split(self.key, n)
                data,clusters = vmap(generate_row)(nkeys)
                return data,clusters
