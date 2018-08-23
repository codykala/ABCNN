# coding=utf-8

import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import save_checkpoint
from utils import generate_plots

# Use GPU if available, otherwise use CPU.
USE_CUDA = torch.cuda.is_available()

PROGRESS_MSG = \
"""
Macro-level accuracy: {}
Macro-level precision: {}
Macro-level recall: {}
Macro-level F1: {}
"""

def train(model, loss_fn, optimizer, history, trainset, valset, config):
    """ Trains the model by optimizing with respect to the given loss
        function using the given optimizer.

        Args:
            model: torch.nn.Module 
                Defines the model.
            loss_fn: torch.nn.Module 
                Defines the loss function.
            optimizer: torch.optim.optimizer 
                Defines the optimizer.
            history: dict
                Contains histories of desired run metrics.
            trainset: torch.utils.data.Dataset 
                Contains the training data.
            valset: torch.utils.data.Dataset 
                Contains the validation data.
            config: dict
                Configures the training loop. Contains the following keys:

                batch_size: int
                    The number of examples to process per batch.
                    Default value is 64.
                start_epoch: int
                    The epoch to start on for training.
                    Default value is 1.
                num_epochs: int
                    How many epochs to train the model.
                    Default value is 20.
                log_every: int
                    How often to save model checkpoint. 
                    To turn off logging, set this value to 0.
                    Default value is 5.
                plot_every: int
                    How often to generate plots.
                    To turn off plotting, set this value to 0.
                    Default value is 5.
                num_workers: int
                    How many works to assign to the DataLoader.
                    Default value is 4.
                verbose: boolean
                    Whether or not to print results to console during training.
                    Progress bar is still included. Default value is False.

        Returns:
            model: a torch.nn.Module defining the trained model.
    """

    # Get keyword parameter values
    batch_size = config.get("batch_size", 20)
    start_epoch = config.get("start_epoch", 1)
    num_epochs = config.get("num_epochs", 20)
    log_every = config.get("log_every", 5)
    plot_every = config.get("plot_every", 5)
    num_workers = config.get("num_workers", 4)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    verbose = config.get("verbose", False)
    gamma = config.get("gamma", 0.1)

    # Learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # Use the f1 score to determine best checkpoint
    best_val_f1 = 0 

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs", position=0):

        # Process training dataset
        model, train_results, train_cm = \
            process_batches(model, trainset, loss_fn, batch_size, num_workers, 
                            desc="train", optimizer=optimizer, is_training=True)
        if verbose:
            tqdm.write(
                PROGRESS_MSG.format(
                    train_results["accuracy"],
                    train_results["precision"],
                    train_results["recall"],
                    train_results["f1"]
                )
            )
    
        # Process validation dataset
        val_results, val_cm = \
            process_batches(model, valset, loss_fn, batch_size, num_workers, 
                            desc="val", optimizer=optimizer, is_training=False)
        if verbose:
            tqdm.write(
                PROGRESS_MSG.format(
                    val_results["accuracy"],
                    val_results["precision"],
                    val_results["recall"],
                    val_results["f1"]
                )
            )

        # Take step for LR
        scheduler.step()

        # Update run history
        for name, val in train_results.items(): 
            history["train_{}".format(name)].append(val)
        for name, val in val_results.items():
            history["val_{}".format(name)].append(val)

        # Update best checkpoint
        if val_results["f1"] > best_val_f1:
            if verbose:
                tqdm.write("New best checkpoint!")
            best_val_f1 = val_results["f1"]
            filepath = os.path.join(checkpoint_dir, "best_checkpoint")
            save_checkpoint(model, optimizer, history, epoch + 1, filepath)

        # Save checkpoint
        if log_every != 0 and epoch % log_every == 0:
            if verbose:
                tqdm.write("Saving checkpoint...")
            filename = "checkpoint_epoch_{}".format(epoch)
            filepath = os.path.join(checkpoint_dir, filename)
            save_checkpoint(model, optimizer, history, epoch + 1, filepath)

        # Generate plots
        if plot_every != 0 and epoch % plot_every == 0:
            if verbose:
                tqdm.write("Generating plots...")
            generate_plots(history, checkpoint_dir)

    return model


def evaluate(model, dataset, loss_fn, batch_size=64, num_workers=4, desc="eval"):
    """ Simple wrapper function for process_batches to evaluate the model 
        the given dataset. 
    """
    results, cm = process_batches(model, dataset, loss_fn, batch_size, 
                                    num_workers, desc=desc)
    print(
        PROGRESS_MSG.format(
            results["accuracy"],
            results["precision"],
            results["recall"],
            results["f1"]
        )
    )
    return results, cm


def process_batches(model, dataset, loss_fn, batch_size=64, num_workers=4, desc=None, 
                    optimizer=None, is_training=False):
    """ Processes the examples in the dataset in batches. If the dataset is the
        training set, then the model weights will be updated.

        Args:
            model: torch.nn.Module
                Defines the model.
            dataset: torch.utils.data.Dataset
                Contains the examples to process.
            loss_fn: torch.nn.Module
                Defines the loss function. The loss function takes batches
                of scores and targets and returns a batch of losses.
            batch_size: int
                The number of examples to process per batch.
                Default value is 64.
            num_workers: int
                How many workers to assign to the DataLoader.
                Default value is 4.
            desc: string
                Optional, writes a short description at the front of the
                progress bar.
            optimizer: torch.optim.optimizer
                Optional, defines the optimizer. Only required for training.
                By default, this is None.
            is_training: boolean
                Optional, specifies whether or not to update model weights.
                By default, this is False.

        Returns:
            model: torch.nn.Module
                The updated model. Returned only if is_training is True.
            results: dict
                Contains relevant evaluation metrics for the model.
            cm: np.array
                The confusion matrix for the model on the dataset.
    """
    # Switch between training and eval mode
    if is_training:
        model = model.train()
    else:
        model = model.eval()
    
    # Track loss across all batches
    total_loss = 0

    # Track number of correct and incorrect predictions
    actual = []
    predicted = []

    # Process batches
    dataloader = \
        DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                    num_workers=num_workers)
    for features, targets in tqdm(dataloader, desc=desc, position=1):

        # Load batch to GPU
        if USE_CUDA:
            torch.cuda.empty_cache()
            features = features.cuda()
            targets = targets.cuda()

        # Forward pass
        scores = model(features)
        predictions = torch.argmax(scores, dim=1)

        # Store actual and predicted labels
        actual.extend(targets.cpu().tolist())
        predicted.extend(predictions.cpu().tolist())

        # Update loss
        batch_loss = torch.sum(loss_fn(scores, targets))
        total_loss += float(batch_loss)

        # Backward pass
        if is_training:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()     

    # Compute evaluation metrics
    avg_loss = total_loss / len(dataset)
    accuracy = accuracy_score(actual, predicted)  
    precision = precision_score(actual, predicted, average="macro")
    recall = recall_score(actual, predicted, average="macro")
    f1 = f1_score(actual, predicted, average="macro")

    # Generate confusion matrix
    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    # Store the results
    results = {
        "total_loss": total_loss,
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    if is_training:
        return model, results, cm
    return results, cm
