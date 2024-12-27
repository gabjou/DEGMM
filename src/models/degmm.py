import jax
import jax.numpy as jnp
import numpy as np
import jax.numpy as jnp
from flax.training import train_state
import optax
from jax import grad, jit, vmap
from jax.random import PRNGKey, split, normal

from src.utils.functions import log_likelihood_egmm
from src.models.mlp import MLP
import logging
from src.utils.functions import gaussian_density

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DEGMM:
    def __init__(self, D,hidden_dim, M, K, lr=0.001, logger=logger,lambda1=0.1,lambda2=0.005):
        self.D = D
        self.hidden_dim = hidden_dim
        self.M = M
        self.K = K


        self.lr = lr
        self.logger = logger
        self.epoch = -1
        self.loss_value = jnp.inf

        rng = PRNGKey(0)
        self.mlp = MLP(input_dim=D*M, hidden_dim=hidden_dim, K=K)
        self.params = self.mlp.init(rng, jnp.ones((D*M,)))
        self.state = train_state.TrainState.create(
            apply_fn=self.mlp.apply,
            params=self.params,
            tx=optax.adam(self.lr)
        )
        


        # Define the optimizer



        self.parameters =[{
            "epoch": self.epoch,
            'pi': None,
            'mu': None,
            'Sigma': None,
            "loglikelihood": self.loss_value,
            "clusters": None
        }]
        self.lambda1 = lambda1
        self.lambda2 = lambda2


    @property
    def model_parameters(self):
        return self.parameters
    
    @model_parameters.setter
    def model_parameters(self, value):
        self.parameters.append(value)


    def loss_function(self, X,pi, mu, Sigma):
        # Calculate the log-likelihood
        log_lik = self.lambda1 * log_likelihood_egmm(X, pi, mu, Sigma)
        # Penalisation of singular matrices
        penalisation = self.lambda2 * jnp.sum(vmap(lambda k: jnp.sum(jnp.linalg.diagonal(Sigma[k,:,:])))(jnp.arange(self.K)))

        return -log_lik+penalisation  # Negative log-likelihood for minimization


    def fit(self, X, num_epochs=100, epsilon=1e-6, patience=3):
        self.epsilon = epsilon
        self.patience = patience
        patience_iter = 0
        n = X.shape[0]

        @jit
        def train_step(state, X):
            def loss_fn(params):
                output = self.mlp.apply(params, X.reshape(n, self.D * self.M))
                gamma = output
                
                N_k = jnp.sum(gamma, axis=0)
                pi = N_k / n
                mu = vmap(lambda k: jnp.sum(vmap(lambda d: gamma[:,k]*(jnp.sum(X, axis=1)[:,d]))(jnp.arange(self.D)).T,axis=0) / (self.M * N_k[k]))(jnp.arange(self.K))


                Sigma = jnp.zeros((self.K, self.D, self.D))
                for k in range(self.K):
                    weighted_X = jnp.sum(vmap(lambda i:gamma[i,k]*(X[i, :, :] - mu[k,:]).T@(X[i, :, :] - mu[k,:]))(jnp.arange(n)),axis=0)
                    Sigma = Sigma.at[k].set(weighted_X/ (self.M * N_k[k]) + 1e-6 * jnp.eye(self.D))
                loss_value = self.loss_function(X, pi, mu, Sigma)
                return loss_value, (pi, mu, Sigma, gamma)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss_value, (pi, mu, Sigma, gamma)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_value, pi, mu, Sigma, gamma

        for epoch in range(num_epochs):
            self.state, loss_value, pi, mu, Sigma, gamma = train_step(self.state, X)
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value.item()}')
            self.model_parameters = {
                "epoch": epoch,
                'pi': pi,
                'mu': mu,
                'Sigma': Sigma,
                "loglikelihood": loss_value,
                "clusters": jnp.argmax(gamma, axis=1)
            }
            if abs(self.model_parameters[-1]["loglikelihood"] - self.model_parameters[-2]["loglikelihood"]) < self.epsilon:
                patience_iter += 1

            if patience_iter >= self.patience:
                break

    
    def predict(self, X,types="probabilities"):
        # Forward pass through the MLP
        output = self.mlp(X)
        # Convert output to JAX array
        output_jax = jnp.array(output.detach().numpy())
        # Extract the parameters
        pi_pred = output_jax[0]
        mu_pred = output_jax[1]
        Sigma_pred = output_jax[2]
        if types=="probabilities":
            return self.compute_probabilities(X, pi_pred, mu_pred, Sigma_pred)
        else:
            return pi_pred, mu_pred, Sigma_pred
    
    def compute_probabilities(X, pi, mu, Sigma):
        def compute_gamma(x):
            densities = vmap(lambda m, s: gaussian_density(x, m, s))(mu, Sigma)
            weighted_densities = pi * densities
            return weighted_densities / jnp.sum(weighted_densities)
        
        gamma = vmap(compute_gamma)(X)
        return gamma
