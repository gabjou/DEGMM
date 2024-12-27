import numpy as np
import jax.numpy as jnp
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from celluloid import Camera

# create figure object


def plot_cluster(X,clusters,params,palette,ax,name):
    M = X.shape[1]
    X_cluster = X.reshape(-1,2)
    clusters_r = jnp.tile(clusters[:,jnp.newaxis],M)
    clusters_r = clusters_r.reshape(-1)
    # for i, cluster in enumerate(np.unique(clusters)):
    #     X_cluster = X[clusters==cluster,:,:]
    #     X_cluster = X_cluster.reshape(-1,2)
    ax.scatter(X_cluster[:, 0], X_cluster[:, 1], s=10,c=clusters_r, cmap=palette, alpha=0.5)
    ax.scatter(params['mu'][:, 0], params['mu'][:, 1],c=np.unique(clusters), s=100, marker='s',edgecolor='black', linewidth=3, cmap =palette)

        # plot_cov_ellipse(params['Sigma'][i], params['mu'][i], ax=ax, edgecolor=palette[i])


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance matrix (`cov`)
        and mean (`pos`).
        """
        if ax is None:
            ax = plt.gca()

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # Compute the angle of the ellipse
        theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        # Width and height of the ellipse
        width, height = 2 * nstd * np.sqrt(eigvals)

        # Draw the ellipse
        ellipse = Ellipse(xy=pos[:2], width=width, height=height, angle=theta, **kwargs)
        ax.add_patch(ellipse)






def animate_inference(X,model_parameters,frames_list,name, filename='inference_animation.gif', palette=['r', 'g', 'b', 'y', 'c', 'm']):
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    camera = Camera(fig)
    for i in frames_list:

        plot_cluster(X, model_parameters[i]['clusters'], model_parameters[i], palette, axes, name)
        axes.text(0.5, 1.01, f'{name} Inference, Epoch {i+1}', transform=axes.transAxes)
        plt.pause(0.1)
        camera.snap()

    animation = camera.animate()
    animation.save(filename, writer='PillowWriter', fps=2)