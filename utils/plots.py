""" Plots """
# standard libraries
import matplotlib.pyplot as plt


class Plot:
    def __init__(self) -> None:
        pass    
    
    def plot_loss(self, train_loss, val_loss, fig_size = (10, 10)):
        """ plots losses"""
        fig, ax = plt.subplots(fig_size=fig_size)
        ax.plot(train_loss, color = "blue", label = "train loss")
        ax.plot(val_loss, color = "yellow", label = "validation loss")
        ax.set_title("Training & Validation losses")
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss")
        return fig
    
    def plot_pca(self, first_componant, second_componant, labels, third_componant = None, fig_size = (10,10)):
        """ plots pca """
        if third_componant == None :
            fig, ax = plt.subplots(fig_size=fig_size)
            ax.scatter(first_componant, second_componant, c = labels)
            ax.set_title("Images in 2D space")
            plt.show()
        else :
            plt.figure(figsize = fig_size)
            ax = plt.axes(projection='3d')
            ax.scatter3D(first_componant, second_componant, third_componant, c = labels)
            plt.show()
        return fig

    def save_fig(self, fig, fig_name):
        fig.savefig(fig_name)



