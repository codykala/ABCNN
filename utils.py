# coding=utf-8

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend("agg")  

def save_checkpoint(model, optimizer, history, epoch):
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
    pass


def load_checkpoint(checkpoint_path):
    """ Loads a model from a pickled checkpoint file.

        Args:
            checkpoint_path: string
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
    pass
    

def generate_plots(history):
    """ Creates plots of the metrics.
    
        Args:
            history: dict
                Contains histories of desired run metrics.

        Returns:
            None
    """
    pass