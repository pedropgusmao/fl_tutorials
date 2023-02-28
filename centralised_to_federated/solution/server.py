"""Flower server example."""
import argparse
import flwr


# Defined parameters to be passed to
def fit_config(server_round: int):
    config = {
        "batch_size": 32,
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
    # evaluate_fn=get_evaluate_fn(model, args.toy),
    on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config,
)


def main(args):
    flwr.server.start_server(
        server_address="127.0.0.1:8080",
        config=flwr.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Flower Client",
        description="This client trains a CNN on a partition of CIFAR10",
    )
    parser.add_argument("--num_clients", type=int, help="Total Number of clients.")
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=1.0,
        help="Fraction of clients to be sampled for training.",
    )
    args = parser.parse_args()
    main(args)
