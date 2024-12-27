from jax import numpy as jnp
from jax.random import PRNGKey, split, normal
import jax
from src.utils.functions import gaussian_density, log_likelihood_gmm
from jax import vmap

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMM:
    def __init__(self, K, max_iter=100, tol=1e-6,logger=logger):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.pi = None
        self.mu = None
        self.Sigma = None
        self.logger = logger
        self.epoch = -1
        self.loss_value = jnp.inf
        self.parameters =[{
            "epoch": self.epoch,
            'pi': self.pi,
            'mu': self.mu,
            'Sigma': self.Sigma,
            "loglikelihood": self.loss_value
        }]


    @property
    def model_parameters(self):
        return self.parameters
    
    @model_parameters.setter
    def model_parameters(self, value):
        self.parameters.append(value)

    def fit(self, X):
        n, D = X.shape
        key = PRNGKey(0)
        key, subkey = split(key)
        self.mu = normal(subkey, (self.K,D))
        key, subkey = split(key)
        indices = jnp.arange(X.shape[0])
        random_indices = jax.random.choice(subkey, indices, shape=(self.K,), replace=False)
        self.mu = X[random_indices]
        self.Sigma = jnp.array([jnp.eye(D) for _ in range(self.K)])
        self.pi = jnp.ones(self.K) / self.K
        log_lik_old = -jnp.inf


        self.model_parameters = {
                    "epoch": self.epoch,
                    'pi': self.pi,
                    'mu': self.mu,
                    'Sigma': self.Sigma,
                    "loglikelihood": log_lik_old,
                    "clusters": None
                }

        for epoch in range(self.max_iter):
            gamma = self.e_step(X)
            self.m_step(X, gamma)

            log_lik = log_likelihood_gmm(X, self.pi, self.mu, self.Sigma)
            if jnp.abs(log_lik - log_lik_old) < self.tol:
                self.logger.info(f'Converged after {epoch+1} epochs')
                break
            log_lik_old = log_lik
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch+1}, Log-Likelihood: {log_lik}')
            self.model_parameters = {
                    "epoch": self.epoch,
                    'pi': self.pi,
                    'mu': self.mu,
                    'Sigma': self.Sigma,
                    "loglikelihood": log_lik,
                    "clusters": jnp.argmax(gamma, axis=1)
                }

    def e_step(self, X):
        K = self.K
        n = X.shape[0]
        gamma = jnp.zeros((n, K))

        denom = vmap(lambda i:jnp.sum(vmap(lambda k: self.pi[k] * gaussian_density(X[i,:], self.mu[k,:].T, self.Sigma[k,:,:]))(jnp.arange(K)))
                                  )(jnp.arange(n))

        for k in range(K):
            gamma_k = vmap(lambda i: self.pi[k] * gaussian_density(X[i,:], self.mu[k,:].T, self.Sigma[k,:,:]))(jnp.arange(n))
            gamma = gamma.at[:, k].set(gamma_k)
        gamma = gamma / denom[:, jnp.newaxis]
        return gamma

    def m_step(self, X, gamma):
        n, K = gamma.shape
        D = X.shape[1]
        N_k = jnp.sum(gamma, axis=0)
        self.pi = N_k / n
        self.mu = jnp.dot(gamma.T, X) / N_k[:, jnp.newaxis]
        self.Sigma = jnp.zeros((K, D, D))
        for k in range(K):
            weighted_X = jnp.sqrt(gamma[:, k][:, jnp.newaxis]) * (X - self.mu[k])
            self.Sigma = self.Sigma.at[k].set(jnp.dot(weighted_X.T, weighted_X) / N_k[k])

    def predict(self, X):
        gamma = self.e_step(X)
        return jnp.argmax(gamma, axis=1)