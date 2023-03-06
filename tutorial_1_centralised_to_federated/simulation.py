"""Flower server example."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import flwr
import logging
from client import CifarClient
from flwr.server.history import History

from shared.utils import (Net, aggregate_weighted_average, get_device,
                          load_partitioned_data)

DEVICE = get_device()

# Defined parameters to be passed to


def fit_config(server_round: int):
    config = {
        "batch_size": 16,
        "local_epochs": 1,
        "learning_rate": 0.2,
    }
    return config


strategy = flwr.server.strategy.FedAvg(
    min_available_clients=2,
    fraction_fit=1.0,
    min_fit_clients=2,
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
)


def client_fn(cid: str):
    # create a single client instance
    trainloader, testloader, num_examples = load_partitioned_data(
        partitions_root=Path('dataset/lda/2/100000.00/') / f"{cid}", batch_size=16
    )

    # Load model
    model = Net().to(DEVICE).train()

    return CifarClient(model, trainloader, testloader, num_examples)


def main(args, strategy=strategy):
    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False,
                     "logging_level": logging.ERROR, "num_cpus": 8}

    client_resources = {
        "num_cpus": 4,
        "num_gpus": 1.0,
    }

    # start simulation
    flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        client_resources=client_resources,
        config=flwr.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )


def execute(strategy=strategy):
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Flower Server",
            description="This server orchestrates an FL ",
        )
        parser.add_argument("--num_clients", type=int,
                            help="Total number of clients.")
        parser.add_argument(
            "--num_rounds", type=int, default=3, help="Total number of rounds."
        )
        parser.add_argument(
            "--fraction_fit",
            type=float,
            default=1.0,
            help="Fraction of clients to be sampled for training.",
        )

        args = parser.parse_args()
        main(args, strategy=strategy)


execute(strategy)
