"""Flower server example."""
import argparse
import flwr
from shared.utils import aggregate_weighted_average
from tutorial_1_centralised_to_federated.solution.server import main, fit_config
# NOTE: add import here

def execute():
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
        # NOTE: add parameter here

        args = parser.parse_args()
        
        # NOTE: change strategy here
        strategy = flwr.server.strategy.FedAvg(
            min_available_clients=2,
            fraction_fit=1.0,
            min_fit_clients=2,
            fraction_evaluate=1.0,
            min_evaluate_clients=2,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=aggregate_weighted_average,
            # NOTE: add parameter here
        )

        main(args, strategy=strategy)


execute()
