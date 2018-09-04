# coding=utf-8

import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")  

def move_to_device(device, *tensors):
    """ Moves the given modules / tensors to the appropriate device.

        If a device was specified in the training configuration, then
        tensors / modules will be moved to that device. Otherwise, 
        tensors / modules will be moved to any available GPUs. If no
        configured GPUs are available, then tensors / modules will remain
        on the CPU.

        Args:
            tensors: list of tensors / modules
                Contains the tensors / modules we would like to move.
               
        Returns:
            tensors: list of tensors / modules
                Contains the tensors / modules moved to the proper device.
                If only tensor was given as input, then that single tensor
                is returned.
    """
    if device:
        if "cuda" in device:
            torch.cuda.empty_cache()
        tensors = list(map(lambda t: t.to(device=device), tensors))
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        tensors = list(map(lambda t: t.cuda(), tensors))
    else:
        tensors = list(map(lambda t: t.cpu(), tensors))

    return tensors[0] if len(tensors) == 1 else tensors

def save_checkpoint(model, optimizer, history, filepath):
    """ Saves the state of the model to a pickle file so that it can continue 
        to be trained at a later time.

        Args:
            model: torch.nn.Module 
                Defines the model.
            optimizer: torch.optim.optimizer 
                Defines the optimizer.
            history: dict
                Contains histories of desired run metrics.
            filepath
                The path where the checkpoint file will be saved.
        
        Returns:
            None
    """
    # Move everything to CPU so model can be loaded into CPU or
    # GPU next time
    state = (
        model.state_dict(),
        optimizer.state_dict(),
        history,
    )
    torch.save(state, filepath)


def load_checkpoint(filepath, map_location=None):
    """ Loads a model from a pickled checkpoint file.

        Args:
            filepath: string
                The path to the checkpoint file.
            map_location: string or callable
                Specifies where the loaded tensors should be stored.
                If not provided, then the Tensors will be loaded
                to wherever they were loaded when the model was saved.
                Use device tags like "cpu" or "cuda:device_id" (i.e. "cuda:0")
                to specify a device.

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
    state = torch.load(filepath, map_location)
    return state
    

def generate_plots(history, checkpoint_dir):
    """ Creates plots of the metric values stored in history.
    
        Args:
            history: defaultdict
                Contains histories of desired run metrics.
            checkpoint_dir: string
                Where to save the plots

        Returns:
            None
    """
    for metric, vals in history.items():
       time = np.arange(1, len(vals) + 1)
       plt.scatter(time, vals, marker='x', color='red')
       plt.xlabel("time")
       plt.ylabel(metric)
       plt.title("{} vs time".format(metric))
       plot_name = "{}_epoch_{}.png".format(metric, len(vals))
       plot_name = os.path.join(checkpoint_dir, plot_name)
       plt.savefig(plot_name)
       plt.clf() 
