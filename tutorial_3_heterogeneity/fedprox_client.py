from tutorial_1_centralised_to_federated.solution.client import CifarClient, main

import argparse
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


DEVICE = get_device()


def get_fedprox_regulariser(*args, **kwargs):
    # Add global params

    def add_fedprox_to_loss(
        net: Module,
        pre_train_network: Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ):
        return 0

    return add_fedprox_to_loss


class FedProxClient(CifarClient):
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        optimizer = SGD(self.model.parameters(), lr=float(config["learning_rate"]))

        fedprox_regulariser = get_fedprox_regulariser(config["proximal_mu"])
        train_regularised(
            self.model,
            self.trainloader,
            optimizer=optimizer,
            epochs=1,
            device=DEVICE,
            regulariser=fedprox_regulariser,
        )
        return self.get_parameters(config={}), self.num_examples["trainset"], {}


def execute():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Flower Client",
            description="This client trains a CNN on a partition of CIFAR10",
        )
        parser.add_argument("--cid", type=int, help="Client ID.")
        parser.add_argument(
            "--partitions_root",
            type=str,
            help="Directory containing client partitions.",
        )
        args = parser.parse_args()
        main(args, FedProxClient)


execute()
