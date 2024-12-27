import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    input_dim: int
    hidden_dim: int
    K: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        gamma = nn.softmax(nn.Dense(self.K)(x))
        return gamma
