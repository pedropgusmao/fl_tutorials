"""Flower client example using PyTorch for CIFAR-10 image classification."""

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
)

from torch.nn import Module
from torch.utils.data import DataLoader

DEVICE = get_device()


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.eval()
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        optimizer = SGD(self.model.parameters(), lr=float(config["learning_rate"]))
        loss, accuracy = train(
            self.model, self.trainloader, optimizer=optimizer, epochs=1, device=DEVICE
        )
        return (
            self.get_parameters(config={}),
            self.num_examples["trainset"],
            {"accuracy": accuracy},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": accuracy}


def main(args) -> None:
    """Load data, start CifarClient."""

    # Load
    trainloader, testloader, num_examples = load_partitioned_data(
        partitions_root=Path(args.partitions_root) / f"{args.cid}", batch_size=32
    )

    # Load model
    model = Net().to(DEVICE).train()

    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


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
        main(args)


execute()
