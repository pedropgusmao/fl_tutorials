from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
from .common import create_lda_partitions
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from flwr.common import *
from copy import deepcopy

DATA_ROOT = "./dataset"


class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_regularised(
    net: Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,  # pylint: disable=no-member
    epochs: int,
    regulariser: Optional[
        Callable[[Module, Module, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
) -> Tuple[float, float]:
    """Train the network."""
    pre_train_network = deepcopy(net)
    net.to(device)
    net.train()

    pre_train_network.to(device)
    pre_train_network.train()
    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        with tqdm(trainloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                images, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = criterion(outputs, labels)
                if regulariser is not None:
                    loss = loss + regulariser(net, pre_train_network, images, labels)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                correct += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()

                # Get statistics
                running_loss += loss.item()
                total += len(labels)

                tepoch.set_postfix(
                    loss=running_loss / total, accuracy=100.0 * correct / total
                )

    return running_loss / total, correct / total


def train(
    net: Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    device: str,  # pylint: disable=no-member
    epochs: int,
) -> Tuple[float, float]:
    """Train the network."""
    net.to(device)
    net.train()

    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        with tqdm(trainloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                images, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                correct += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()

                # Get statistics
                running_loss += loss.item()
                total += len(labels)

                tepoch.set_postfix(
                    loss=running_loss / total, accuracy=100.0 * correct / total
                )

    return running_loss / total, correct / total


def compute_model_delta(trained_parameters: NDArrays, og_parameters: NDArrays):
    return [np.subtract(x, y) for (x, y) in zip(trained_parameters, og_parameters)]


def compute_norm(update: NDArrays) -> float:
    """Compute the l1 norm of a parameter update with mismatched np array shapes, to be used in clipping"""
    flat_update = update[0]
    for i in range(1, len(update)):
        flat_update = np.append(flat_update, update[i])  # type: ignore
    summed_update = np.abs(flat_update)
    norm_sum = np.sum(summed_update)
    norm = np.sqrt(norm_sum)
    return norm


def test(
    net: Module,
    testloader: DataLoader,
    device: str,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss, total = 0, 0.0, 0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            tepoch.set_description(f"Validation set")
            for data in tepoch:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                total += len(outputs)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                correct += (predicted == labels).sum().item()
    return loss / total, correct / total


def load_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load CIFAR-10 (training and test set)."""
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


def load_partitioned_data(
    partitions_root: Path, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, Dict]:
    transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    with open(partitions_root / f"train.pickle", "rb") as f:
        np_trainset = pickle.load(f)
    trainset = CustomTensorDataset(
        tensors=[
            Tensor(np.transpose(np_trainset[0], (0, 3, 1, 2))),
            Tensor(np_trainset[1]).to(torch.int64),
        ],
        transform=transform,
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    with open(partitions_root / f"test.pickle", "rb") as f:
        np_testset = pickle.load(f)
    testset = CustomTensorDataset(
        tensors=[
            Tensor(np.transpose(np_testset[0], (0, 3, 1, 2))),
            Tensor(np_testset[1]).to(torch.int64),
        ],
        transform=transform,
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index] / 255

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device


def create_lda_cifar10_partitions(num_partitions: int, concentration: float) -> bool:
    partition_root = (
        Path(DATA_ROOT) / "lda" / f"{num_partitions}" / f"{concentration:.2f}"
    )

    if partition_root.is_dir():
        print("Partitions already exist. Delete if necessary.")
        return True

    trainset = CIFAR10(
        DATA_ROOT,
        train=True,
        download=True,
    )
    testset = CIFAR10(
        DATA_ROOT,
        train=False,
        download=True,
    )

    x = np.array(trainset.data)
    y = np.array(trainset.targets)

    train_clients_partitions, train_dists = create_lda_partitions(
        dataset=(x, y),
        dirichlet_dist=None,
        num_partitions=num_partitions,
        concentration=concentration,
        accept_imbalanced=True,
    )
    x = np.array(testset.data)
    y = np.array(testset.targets)
    test_clients_partitions, dist = create_lda_partitions(
        dataset=(x, y),
        dirichlet_dist=train_dists,
        num_partitions=num_partitions,
        concentration=concentration,
        accept_imbalanced=True,
    )

    for cid in range(num_partitions):
        # Create partitions
        partition_cid = partition_root / f"{cid}"
        partition_cid.mkdir(parents=True)
        train_file = partition_cid / "train.pickle"
        train_partition = train_clients_partitions[cid]
        with open(train_file, "wb") as f:
            pickle.dump(train_partition, f)

        test_file = partition_cid / "test.pickle"
        test_partition = test_clients_partitions[cid]
        with open(test_file, "wb") as f:
            pickle.dump(test_partition, f)

    return True
