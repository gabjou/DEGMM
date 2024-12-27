# DEGMM
Deep Exchangeable Gaussian Mixture Model
## Overview
DEGMM is a deep learning model designed for clustering and density estimation. It leverages the power of Gaussian Mixture Models (GMM) and deep neural networks to provide a flexible and scalable solution.


<figure>
  <img
  src="/figures/figures/example.png"
  >
  <em>Example of Bivariate gaussian exchangeable ensemble following a K=3 components exchangeable Guassian mixture.</em>
</figure>


## Features
- Combines Gaussian mixture model with deep learning
- Jax and Flax implementation
- Deep learning inference using Exchangeable Gaussian mixture loglikelihood
- Flexible architecture

## Installation
To install DEGMM, clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/DEGMM.git
cd DEGMM
pip install -r requirements.txt
```

## Usage
Here's a basic example of how to use DEGMM:
```shell
python main.py --n 1000 --M 10 --D 2 --K 3 --num_epochs 1000 --seed 0
```

## Contributing
We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please open an issue on GitHub or contact us at support@degmm.com.