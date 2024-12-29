# DEGMM
Deep Exchangeable Gaussian Mixture Model (DEGMM)
## Overview
DEGMM is a deep learning model designed for clustering and density estimation of an ensemble dataset distributed following an exchangeable gaussian mixture. In this repository, the traditional Expectation-Maximisation (EM) algorithm used to infer a Gaussian Mixture Models (GMM) is compared to a Deep learning inference (DGMM) and the novel DEGMM. DGMM and DEGMM rely on a similar architecture inspired of this two papers [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-) and [Gaussian mixture models for clustering and calibration of ensemble weather forecasts](https://www.researchgate.net/profile/Goulven-Monnier/publication/358989436_Gaussian_mixture_models_for_clustering_and_calibration_of_ensemble_weather_forecasts/links/638789c0bbdef30dc9877e90/Gaussian-mixture-models-for-clustering-and-calibration-of-ensemble-weather-forecasts.pdf). In DGMM and DEGMM, a traditional multi-layer perceptron (MLP) takes all ensemble members as inputs and ouput gamma, the cluster probabilities. The MLP model resolves the expectation step of the traditional EM inference. Using the ouput of the MLP, the same equations of the maximisation step is used to update parameters of the GMM and exchangeable GMM. The loglikelihoods of the GMM and exchangeable GMM are used as objective function to optimize parameters of the DGMM and DEGMM models. This work leverages the power of GMM, exchangeability and deep neural networks using jax and flax to provide a flexible and scalable solution for exchangeable ensemble clustering. This work is 

<figure>
  <img
  src="/figures/example.png"
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
- n: Size of the dataset (number of samples)
- M: Size of the ensemble (number of ensemble members)
- D: Space dimension (D=2 generate bivariate samples)
- K: Number of mixture compnents (Number of clusters)
<figure>
  <img
  src="/figures/results.png"
  >
  <em>Results of the DEGMM clustering for bivariate exchangeable ensemble data of M=10 members, with n=10000 and K=3 gaussian components. DEGMM results are compared to a simple Gaussian mixture model using the Expectation-Maximisation algorithm. Square symbolizes the cluster mean.</em>
</figure>



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please open an issue on GitHub or contact us at gabrijou@gmail.com.