import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.random import PRNGKey, split, normal
from jax import vmap

# Define the Gaussian density function
def gaussian_density(x, mu, Sigma):
    return multivariate_normal.pdf(x, mean=mu, cov=Sigma)

# Define the joint PDF for the Exchangeable Gaussian Mixture Model
def joint_pdf_egmm(x, pi, mu, Sigma):
    # jax.debug.print("x: {}", x)

    K = len(pi)
    M = x.shape[0]
    pdf_value = 0.0
    for k in range(K):
        product = jnp.prod(vmap(lambda m: gaussian_density(x[m,:], mu[k,:].T, Sigma[k,:,:]))(jnp.arange(M)))
        
        pdf_value += pi[k] * product
    return pdf_value

# Define the log-likelihood function of the Exchangeable Gaussian Mixture Model
def log_likelihood_egmm(X, pi, mu, Sigma):

    # jax.debug.print("pi: {}",pi)
    # jax.debug.print("mu: {}", mu)
    # jax.debug.print("Sigma: {}", Sigma)


    log_pdf_egmm = vmap(lambda x: jnp.log(joint_pdf_egmm(x, pi, mu, Sigma)))(X)
    return jnp.sum(log_pdf_egmm)


# Define the PDF for the Gaussian mixture model
def pdf_gmm(x, pi, mu, Sigma):
    K = len(pi)
    pdf_value = 0.0
    for k in range(K):
        pdf_value += pi[k] * gaussian_density(x, mu[k], Sigma[k])
    return pdf_value

# Define the log-likelihood function of the Gaussian mixture model
def log_likelihood_gmm(X, pi, mu, Sigma):
    log_pdf_gmm = vmap(lambda x: jnp.log(pdf_gmm(x, pi, mu, Sigma)))
    return jnp.sum(log_pdf_gmm(X))
