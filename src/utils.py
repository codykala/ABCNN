# coding=utf-8

import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")  

def abcnn_model_loader(filepath, model, optimizer):
    """ Helper function to load a pre-trained ABCNN model from a model
        checkpoint file.

        Args:
            filepath: string
                The path to the ABCNN model checkpoint file.
            model: ABCNN model
                An instance of the ABCNN model.
            optimizer: optim.Optimizer
                An instance of the optimizer.

        Returns:
            model: ABCNN model
                The pre-trained ABCNN model.
            optimizer: optim.Optimizer
                The optimizer used to train the ABCNN model.
    """
    state = load_checkpoint(filepath)
    new_model_dict, new_optim_dict, _, _ = state
    
    # Overwrite the model state
    new_model_dict = {
        k: v for k, v in new_model_dict.items()
        if k != "embeddings.weight" # ignore embedding layer
    }
    model_dict = model.state_dict()
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    # Overwrite the optimizer state
    optimizer.load_state_dict(new_optim_dict)
    
    return model, optimizer


def freeze_weights(pretrained_model):
    """ Creates a copy of the pre-trained model with its conv-pool layer
        weights frozen. This allow us to learn only the weights in the
        fully connected layer.

        Args:
            pretrained_model: Model module
                The pre-trained model we want to use.

        Returns:
            model: Model module
                The model we want to train. The weights for its conv-pool
                layers are identical to the pre-trained model's weights.
    """
    model = copy.deepcopy(pretrained_model)
    for layer in model.layers:
        for param in layer.parameters():
            param.requires_grad = False
    return model


def save_checkpoint(model, optimizer, history, epoch, filepath):
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
        epoch
    )
    torch.save(state, filepath)


def load_checkpoint(filename, map_location=None):
    """ Loads a model from a pickled checkpoint file.

        Args:
            filename: string
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
    state = torch.load(filename, map_location)
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
