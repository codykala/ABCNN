# coding=utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")  

def save_checkpoint(model, optimizer, history, epoch, filename):
    """ Saves the state of the model to a pickle file so that it can continue 
        to be trained at a later time.

        Args:
            model: torch.nn.Module 
                Defines the model.
            optimizer: torch.optim.optimizer 
                Defines the optimizer.
            history: dict
                Contains histories of desired run metrics.
            epoch: int
                The current epoch number.
        
        Returns:
            None
    """
    state = (model.state_dict(), optimizer.state_dict(), history, epoch)
    torch.save(state, filename)


def load_checkpoint(filename):
    """ Loads a model from a pickled checkpoint file.

        Args:
            filename: string
                The path to the checkpoint file.

        Returns
            model: torch.nn.Module
                Defines the model.
            optimizer: torch.optim.optimizer
                Defines the optimizer.
            history: dict
                Contains histories of desired run metrics.
            epoch: int
                The current epoch number.
    """
    state = torch.load(filename)
    return state
    

def generate_plots(history):
    """ Creates plots of the metric values stored in history.
    
        Args:
            history: defaultdict
                Contains histories of desired run metrics.

        Returns:
            None
    """
    for metric, vals in history.items():
       time = np.arange(1, len(vals) + 1)
       plt.scatter(time, vals, marker='x', color='red')
       plt.xlabel("time")
       plt.ylabel(metric)
       plt.title("{} vs time".format(metric))
       plt.savefig("{}_epoch_{}.png".format(metric, len(vals)))
       plt.clf() 
