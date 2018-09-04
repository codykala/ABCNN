# coding=utf-8

import copy
import os
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from string import Template
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange

import trainer.utils

PROGRESS_MSG = Template(
    "Macro-level accuracy: ${accuracy}\n"
    "Macro-level precision: ${precision}\n"
    "Macro-level recall: ${recall}\n"
    "Macro-level f1: ${f1}"
)

class MulticlassClassifierTrainer(object):
    """ This class defines an API for training and evaluating Multiclass
        Classifiers built using PyTorch. """

    def __init__(self, config):
        """ Initializes the MulticlassClassifierTrainer. 

            `config` should be a dictionary with the following key-value
            pairs (all keys are strings):

                batch_size: int
                    The number of examples to process per batch.
                num_epochs: int
                    The number of complete loops over the training set.
                log_every: int
                    Controls how often to log training and validation results.
                num_workers: int
                    Controls how many worker processes to use for training.
                checkpoint_dir: string
                    Specifies the directory where checkpoint files and plots
                    will be saved.
                verbose: bool
                    Determines the verbosity of the output during training and
                    evaluation.
                device: string
                    Specifies the device to use for training and evaluation.
                    To use the CPU, this should be "cpu". To use a configured
                    GPU, use "cuda:<num>". For example, on a computer with 2
                    GPUs, use "cuda:0" to train on the first GPU and "cuda:1"
                    to train on the second GPU.

            Args:
                config: dict
                    The configuration to use for training and evaluation.
        
            Returns:
                None
        """
        # Keep track of current model and best model
        self._best_model = None
        self._model = None
        self._history = None

        # Store train configuration attributes directly for easy access
        for attr, val in config.items():
            setattr(self, attr, val)

    def train(self, 
              loss_fn,
              model,
              optimizer,
              trainset,
              scheduler=None,
              valset=None):
        """ Trains the model on the given training set. If a validation set
            is provided, then the model is evaluated on the validation set

            If only a training set is provided, then the best model will
            be determined by the f1 score on the training set.

            If both a training set and a validation set are provided, then
            the best model will be determined by the f1 score on the
            validation set.

            Args:
                loss_fn: nn.Loss
                    The loss function to use for training.
                model: nn.Module
                    The model to train.
                optim: optim.Optimizer
                    The optimizer to use for training.
                trainset: Dataset
                    Contains the training examples.
                scheduler: optim.Optimizer.lr_scheduler
                    Optional, the learning rate scheduler to use for training. 
                    To skip using a learning rate scheduler, pass None for this
                    argument.
                valset: Dataset
                    Optional, contains the validation examples. To skip using
                    a validatio nset, pass None for this argument.
            
            Returns:
                None
        """
        # Setup
        self._loss_fn = loss_fn
        self._model = model
        self._best_model = copy.deepcopy(model)
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Training loop
        self._best_f1 = 0
        self._history = defaultdict(list)
        for epoch in trange(self.num_epochs, desc="epochs", position=0):

            # Process training set
            train_results, _ = self._process(trainset, False, True, desc="train")
            if self.verbose:
                tqdm.write(PROGRESS_MSG.substitute(train_results))

            # Process validation set, if provided
            if valset:
                val_results, _ = self._process(valset, False, False, desc="val")
                if self.verbose:
                    tqdm.write(PROGRESS_MSG.substitute(val_results))

            # Take step for LR scheduler
            if self._scheduler:
                self._scheduler.step()

            # Update run history
            for name, val in train_results.items():
                self._history["train_{}".format(name)].append(val)
            if val_results:
                for name, val in val_results.items():
                    self._history["val_{}".format(name)].append(val)

            # Update best model
            if valset:
                self._update_best_model(val_results)
            else:
                self._update_best_model(train_results)
        
            # Save checkpoint and plots
            if self.log_every != 0 and epoch % self.log_every == 0:
                if self.verbose:
                    tqdm.write("Saving checkpoint...")
                filename = "checkpoint_epoch_{}".format(epoch)
                filepath = os.path.join(self.checkpoint_dir, filename)
                self._save_checkpoint(filepath)
                # self._save_plots()

    def predict(self, dataset):
        """ Processes the examples in the dataset for evaluation and
            prediction.

            Args:
                dataset: Dataset
                    Contains the examples and their labels.

            Returns:
                results: dict
                    Contains evaluation metrics for the model on the given
                    dataset.
                predicted: list of int
                    Contains the predictions of the examples in the dataset.
        """
        return self._process(dataset, True, False, desc="predicting")

    def _process(self, 
                 dataset, 
                 use_best,
                 is_training,
                 desc=None):
        """ Processes the examples in the dataset.

            Args:
                dataset: Dataset
                    Contains the examples and their labels.
                use_best: bool 
                    Specifies whether to use the current or best model
                    for processing examples. The best model should be used
                    for evaluation or prediction, and the current model
                    should be used for training and validation.
                is_training: bool
                    Specifies whether or not to update model weights.
                    Model weights should only be updated during training.
                    During validation, evaluation, or prediction, this
                    should be False.
                desc: string
                    Optional, write a short description at the front
                    of the progress bar.

            Returns:
                results: dict
                    Contains evaluation metrics for the model on the given
                    dataset.
                preds: list of int
                    Contains the predictions of the examples in the dataset.
        """
        # Get the appropriate model for processing
        model = self._best_model if use_best else self._model
        model = model.train() if is_training else model.eval()

        # Process batches
        actual = []
        predicted = []
        total_loss = 0
        dataloader = \
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        for features, labels in tqdm(dataloader, desc=desc, position=1):
            
            # Load tensors to correct device
            features, labels = self._move_to_device(features, labels)

            # Forward pass
            scores = self._model(features)
            preds = torch.argmax(scores, dim=1)

            # Store actual and predicted labels
            actual.extend(labels.cpu().tolist())
            predicted.extend(preds.cpu().tolist())

            # Update loss
            batch_loss = torch.sum(self._loss_fn(scores, labels))
            total_loss += batch_loss

            # Backward pass
            if is_training:
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

        # Compute evaluation metrics
        avg_loss = total_loss / len(dataset)
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(actual, predicted, average="macro")
        recall = recall_score(actual, predicted, average="macro")
        f1 = f1_score(actual, predicted, average="macro")

        # Return results
        results = {
            "total_loss": total_loss,
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return results, predicted

    def _move_to_device(self, *tensors):
        """ Moves the given modules / tensors to the appropriate device.

            If a device was specified in the training configuration, then
            tensors / modules will be moved to that device. Otherwise, 
            tensors / modules will be moved to any available configured GPUs.
            If no configured GPUs are available, then tensors will remain
            on the CPU.

            Args:
                tensors: list of tensors / modules
                    Contains the tensors / modules we would like to move.

            Returns:
                tensors: list of tensors / modules
                    Contains the tensors / modules moved to the proper device.
        """
        return trainer.utils.move_to_device(self.device, *tensors)

    def _update_best_model(self, results):
        """ Helper function to update the best model observed.

            Args:
                results: dict
                    Contains the results of performance metrics from the
                    current epoch.

            Returns:
                None
        """
        if results["f1"] > self._best_f1:
            if self.verbose:
                tqdm.write("New best checkpoint!")
            self._best_model = copy.deepcopy(self._model)
            self._best_f1 = results["f1"]
            filepath = os.path.join(self.checkpoint_dir, "best_checkpoint")
            self._save_checkpoint(filepath)

    def _save_checkpoint(self, filepath):
        """ Saves a checkpoint of the model at the given filepath.

            The checkpoint saves the following information to a file:

                - the current state of the model
                 the current state of the optimizer
                - the current history of the model
            
            Args:
                filepath: string
                    The path to the model checkpoint.

            Returns:
                None
        """
        trainer.utils.save_checkpoint(
            self._model, 
            self._optimizer, 
            self._history, 
            filepath
        )

    def _save_plots(self):
        """ Saves plots of the model's metric history.

            The plots will be saved to the checkpoint directory.

            Args:
                None

            Returns:
                None
        """
        trainer.utils.generate_plots(
            self._history, 
            self.checkpoint_dir
        )
