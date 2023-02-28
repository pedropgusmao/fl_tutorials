import argparse
from torchvision.datasets import CIFAR10
from utils import create_lda_cifar10_partitions

DATA_ROOT = "./dataset"


def main(args: argparse.Namespace):
    print(
        f"Creating {args.num_partitions} with LDA concentration (alpha) of {args.alpha:.2f} ."
    )
    create_lda_cifar10_partitions(
        num_partitions=args.num_partitions, concentration=args.alpha
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Latent Dirichlet Allocation (LDA) Partitioning for CIFAR10",
        description="Partitions the original CIFAR10 dataset into `num_partitions` partitions using concentration `alpha`.",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=2,
        help="Number of partitions to be created.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=100000,
        help="Concentration value for LDA. This is a number in the range (0,+Inf)",
    )
    args = parser.parse_args()
    main(args)
