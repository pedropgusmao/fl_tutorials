from tutorial_1_centralised_to_federated.solution.server import main
import argparse
from tutorial_2_building_a_strategy.solution.strategy import FedAvgLearningRate


def execute():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Flower Server",
            description="This server orchestrates an FL ",
        )
        parser.add_argument("--num_clients", type=int, help="Total number of clients.")
        parser.add_argument(
            "--num_rounds", type=int, default=3, help="Total number of rounds."
        )
        parser.add_argument(
            "--fraction_fit",
            type=float,
            default=1.0,
            help="Fraction of clients to be sampled for training.",
        )
        # Add argument for proximal mu
        parser.add_argument()

        args = parser.parse_args()

        # Add new config to use proximal mu
        def fit_config(server_round: int):
            config = {}
            return config

        strategy = FedAvgLearningRate(
            min_available_clients=2,
            fraction_fit=1.0,
            min_fit_clients=2,
            fraction_evaluate=1.0,
            min_evaluate_clients=2,
            server_learning_rate=1.0,
            # evaluate_fn=get_evaluate_fn(model, args.toy),
            on_fit_config_fn=fit_config,
            # on_evaluate_config_fn=evaluate_config,
        )
        main(args, strategy=strategy)


execute()
