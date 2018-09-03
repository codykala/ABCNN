# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim

def loss_fn_factory(config):
    """ A convenience function that initializes some of the common loss 
        functions supported by PyTorch.

        Includes:
            - l1
            - mean squared error
            - binary cross entropy
            - binary cross entropy with logits
            - cross entropy
            - negative log likelihood
            - kullback-leibler divergence

        Args:
            config: dict
                Contains the parameters needed to initialize the loss function.

        Returns:
            loss_fn: nn.Loss
                The loss function.

        Raises:
            ValueError
    """
    # Get all of the possible arguments we might need
    ignore_index = config.get("ignore_index", -100)
    pos_weight = config.get("pos_weight", None)
    reduce = config.get("reduce", None)
    reduction = config.get("reduction", "elementwise_mean")
    size_average = config.get("size_average", None)
    weight = config.get("weight", None)
    
    if config["type"] == "l1":
        return nn.L1Loss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
    elif config["type"] == "mean squared error":
        return nn.MSELoss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
    elif config["type"] == "cross entropy":
        return nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction
        )
    elif config["type"] == "negative log likelihood":
        return nn.NLLLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce
        )
    elif config["type"] == "kullback-leibler divergence":
        return nn.KLDivLoss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
    elif config["type"] == "binary cross entropy":
        return nn.BCELoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
    elif config["type"] == "binary cross entropy with logits":
        return nn.BCEWithLogitsLoss(
            weight=None,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight
        )
    else:
        raise ValueError("Unrecognized loss_fn type.")

def optimizer_factory(config, params):
    """ A convenience function that initializes some of the common optimizers
        supported by PyTorch.
        
        Includes:
            - adadelta
            - adagrad
            - adam
            - adamax
            - rmsprop
            - sgd

        Args:
            config: dict
                Contains the parameters needed to initialize the optimizer,
                such as the learning rate, weight decay, etc.
            params: iterable
                An iterable of parameters to optimize or dicts defining
                parameter groups.

        Returns:
            optim: optim.Optimizer
                An optimizer object
    """
    if config["type"] == "adadelta":
        return optim.Adadelta(
            params,
            lr=config.get("lr", 1.0),
            rho=config.get("rho", 0.9),
            eps=config.get("eps", 1e-6),
            weight_decay=config.get("weight_decay", 0)
        )
    elif config["type"] == "adagrad":
        return optim.Adagrad(
            params,
            lr=config.get("lr", 0.01),
            lr_decay=config.get("lr_decay", 0),
            weight_decay=config.get("weight_decay", 0),
            initial_accumulator_value=config.get("initial_accumulator_value", 0)
        )
    elif config["type"] == "adam":
        return optim.Adam(
            params,
            lr=config.get("lr", 0.001),
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0),
            amsgrad=config.get("amsgrad", False)
        )
    elif config["type"] == "adamax":
        return optim.Adamax(
            params,
            lr=config.get("lr", 0.002),
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0)
        )
    elif config["type"] == "rmsprop":
        return optim.RMSProp(
            params, 
            lr=config.get("lr", 0.01),
            alpha=config.get("alpha", 0.99),
            eps=config.get("eps", 1e-8), 
            weight_decay=config.get("weight_decay", 0),
            momentum=config.get("momentum", 0),
            centered=config.get("centered", False)
        )
    elif config["type"] == "sgd":
        return optim.SGD(
            params,
            lr=config.get("lr", 0.001),
            momentum=config.get("momentum", 0),
            dampening=config.get("dampening", 0),
            weight_decay=config.get("weight_decay", 0),
            nesterov=config.get("nesterov", False)
        )
    else:
        raise ValueError("Unrecognized optimizer type.")


def scheduler_factory(config, optimizer):
    """ Initializes a learning rate scheduler using the given configuration. """
    if config["type"] == "lambda":
        return optim.lr_scheduler.LambdaLR(optimizer, **config)
    elif config["type"] == "step":
        return optim.lr_scheduler.StepLR(optimizer, **config)
    elif config["type"] == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, **config)
    elif config["type"] == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **config)
    else:
        raise ValueError("Unrecognized scheduler type")

