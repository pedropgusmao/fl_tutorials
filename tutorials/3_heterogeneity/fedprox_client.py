from centralised_to_federated.solution.client import CifarClient

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

import torch
from torch.optim import SGD
from shared.utils import (
    Net,
    test,
    train,
    load_partitioned_data,
    get_device,
    train_regularised,
)

from torch.nn import Module
from torch.utils.data import DataLoader

from torch.nn import Optimizer
import torch.nn as nn
import tqdm

DEVICE = get_device()


def get_fedprox_regulariser():


    def add_fedprox_to_loss(
        net: Module,
        pre_train_network: Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ):
        pass

    return add_fedprox_to_loss


class FedProxClient(CifarClient):
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        optimizer = SGD(self.model.parameters(), lr=float(config["learning_rate"]))
       

        # Add regulariser
        fedprox_regulariser = lambda p: 0

        train_regularised(
            self.model,
            self.trainloader,
            optimizer=optimizer,
            epochs=1,
            device=DEVICE,
            regulariser=fedprox_regulariser,
        )
        return self.get_parameters(config={}), self.num_examples["trainset"], {}
