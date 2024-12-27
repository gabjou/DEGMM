import jax.numpy as jnp
import jax.random as random
from src.data.generator import DataGenerator
from src.models.gmm import GMM
from src.models.degmm import DEGMM
from src.utils.plots import plot_cluster, animate_inference
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix,accuracy_score


def main(args):
    n = args.n
    M = args.M
    D = args.D
    K = args.K
    num_epochs = args.num_epochs
    seed = args.seed

    
    ################################## Generate data #################################
    key = random.key(seed)
    seedarray = random.split(key, K)

    mus = jnp.array([random.normal(seedarray[k], shape=(D,)) for k in range(K)])
    Sigmas = jnp.array([jnp.diag(jnp.abs(random.normal(seedarray[k], shape=(D,)))) for k in range(K)])
    pi = jnp.abs(random.dirichlet(key, jnp.ones(K)))
    true_params ={
        "mu":mus,
        "Sigma":Sigmas,
        "pi":pi
    }
    # Initialize the DataGenerator
    data_generator = DataGenerator(K, M, mus, Sigmas, pi)

    # Generate data
    X,Z = data_generator.generate_data(n)



    ################################## Fit the models ##################################
    # Initialize the GMM model
    gmm = GMM(K=K)

    # Fit the GMM model on one ensemble member
    timestart = time.time()
    X_member = X[:,random.randint(key,shape=1,minval=0,maxval=M), :].reshape(n, D)
    gmm.fit(X_member)
    timeend = time.time()
    print(f"Time taken to fit GMM: {timeend - timestart} seconds")

    # Initialize the DEGMM model
    degmm = DEGMM(D, hidden_dim=32, M=M, K=K)

    # Fit the DEGMM model
    # resize X to (n, D*M)
    timestart = time.time()
    degmm.fit(X,num_epochs=num_epochs)
    timeend = time.time()
    print(f"Time taken to fit GMM: {timeend - timestart} seconds")


    params_gmm = gmm.model_parameters[-1]
    params_degmm = degmm.model_parameters[-1]


    # Compute the confusion matrix
    y_pred_gmm = params_gmm["clusters"]
    confusion_matrix_gmm = confusion_matrix(Z, y_pred_gmm)

    # reorder y_pred_gmm following the closest cluster to the true cluster
    reorderlabels = confusion_matrix_gmm.argmax(axis=0)
    closest_cluster_gmm = y_pred_gmm.copy()
    for k in range(K):
        closest_cluster_gmm = closest_cluster_gmm.at[y_pred_gmm==k].set(reorderlabels[k])
    params_gmm["mu"] = params_gmm["mu"][reorderlabels,:]
    params_gmm["Sigma"] = params_gmm["Sigma"][reorderlabels,:,:]
    params_gmm["pi"] = params_gmm["pi"][reorderlabels]
    accgmm = accuracy_score(Z, closest_cluster_gmm)


    y_pred_degmm = params_degmm["clusters"]
    confusion_matrix_degmm = confusion_matrix(Z, y_pred_degmm)

    # reorder y_pred_degmm following the closest cluster to the true cluster
    reorderlabels = confusion_matrix_degmm.argmax(axis=0)
    closest_cluster_degmm = y_pred_degmm.copy()
    for k in range(K):
        closest_cluster_degmm = closest_cluster_degmm.at[y_pred_degmm==k].set(reorderlabels[k])
    params_degmm["mu"] = params_degmm["mu"][confusion_matrix_degmm.argmax(axis=1),:]
    params_degmm["Sigma"] = params_degmm["Sigma"][confusion_matrix_degmm.argmax(axis=1),:,:]
    params_degmm["pi"] = params_degmm["pi"][confusion_matrix_degmm.argmax(axis=1)]
    accdegmm = accuracy_score(Z, closest_cluster_degmm)


    ################################## Plot the results ##################################
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    palette = plt.cm.get_cmap('tab10', K)
    # Plot the true clusters
    plot_cluster(X, Z, true_params, palette, ax[0], "True")
    ax[0].set_title(f'True Clusters')
    ax[0].set_xlabel(f'X1')
    ax[0].set_xlabel(f'X2')

    # Plot the results of gmm
    plot_cluster(X, closest_cluster_gmm, params_gmm, palette, ax[1], "GMM")
    ax[1].set_title(f'GMM Clusters ACC={accgmm:.2f}')
    ax[1].set_xlabel(f'X1')
    ax[1].set_xlabel(f'X2')

    # Plot the results of degmm
    plot_cluster(X, closest_cluster_degmm, params_degmm, palette, ax[2], "DEGMM")
    ax[2].set_title(f'DEGMM Clusters ACC={accdegmm:.2f}')
    ax[2].set_xlabel(f'X1')
    ax[2].set_xlabel(f'X2')

    # plt.show()
    fig.tight_layout()
    fig.savefig('./figures/results.png')

    frames_list = [i for i in range(1, num_epochs, 100)]
    # Animate the inference process DEGMM
    animate_inference(X,degmm.model_parameters,frames_list,"DEGMM",filename='./figures/inference_degmm.gif', palette=palette)






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DEGMM on generated data.")
    parser.add_argument("--n", type=int, default=1000, help="Number of data points to generate.")
    parser.add_argument("--M", type=int, default=10, help="Ensemble member size.")
    parser.add_argument("--D", type=int, default=2, help="Dimension of the Gaussian distribution.")
    parser.add_argument("--K", type=int, default=3, help="Number of clusters.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train DEGMM.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    
    args = parser.parse_args()
    main(args)