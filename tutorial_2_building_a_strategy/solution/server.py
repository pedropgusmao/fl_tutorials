from tutorial_1_centralised_to_federated.solution.server import main, fit_config
import argparse
from strategy import FedAvgLearningRate  # <--- NEW
from shared.utils import aggregate_weighted_average


def execute():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Flower Server using FedAvgLearningRate",
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
        # ====== NEW ======
        parser.add_argument(
            "--server_learning_rate",
            type=float,
            default=0.5,
            help="Server learning rate.",
        )
        # =================
        args = parser.parse_args()

        strategy = FedAvgLearningRate(  # <--- NEW
            min_available_clients=2,
            fraction_fit=1.0,
            min_fit_clients=2,
            fraction_evaluate=1.0,
            min_evaluate_clients=2,
            server_learning_rate=args.server_learning_rate,  # <--- NEW
            # evaluate_fn=get_evaluate_fn(model, args.toy),
            on_fit_config_fn=fit_config,
            # on_evaluate_config_fn=evaluate_config,
            evaluate_metrics_aggregation_fn=aggregate_weighted_average,
        )

        main(args, strategy=strategy)


execute()
