# Tutorial 2: building a strategy

In this tutorial, we will explore how to build a Strategy in Flower.
The running example we will use is FedAvg with server learning rate.
We will start almost from scratch to build our Strategy, but we will take advantage of the previous tutorial's implementation.
We will use the Client and some utilities we have already implemented in Tutorial 1.

## Steps of the Tutorial

In this section, the steps of the tutorial are described.

### Step 0: Slides

First slide of the presentation.

### Step 1: Create an empty strategy file

First, we need to create a `strategy.py` file in the main folder of the tutorial.
We want to import some packages from Flower, so we need to add the following lines to the top of the file:

```python
from logging import INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import NDArrays, MetricsAggregationFn, FitRes, FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from shared.utils import compute_model_delta, compute_norm
```

### Step 2: Create a strategy object

Now, we will create a Strategy object that inherits from the `FedAvg` class.
We may want to have a look at the [source](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/strategy.py) of the `Strategy` class or the [source](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py) of the `FedAvg` class to see what methods we can override.
Thus, we can take advantage of most of the methods of the `FedAvg` class and just override only those that are relevant to us.
Let's create the class and write the `__init__` method:

```python
class FedAvgLearningRate(FedAvg):
    """FedAvg with learning rate."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0, # <--- NEW
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.server_learning_rate = server_learning_rate # <--- NEW
```

Note what we added here: a new argument `server_learning_rate` to the `__init__` method.
We will now override two methods to use this new argument.

### Step 3: Override the `configure_fit` method

The first method we want to override is the `configure_fit` method.
We don't need to change much of it, indeed, we will call its parent method from `FedAvg`.
The relevant task we want to implement here is saving the global model parameters.
We will do this here because this method is called before launching the training of each round.

```python
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.previous_parameters = parameters # <--- NEW
        return super().configure_fit(server_round, parameters, client_manager)
```

### Step 4: Override the `aggregate_fit` method

The most relevant method to our running example is the `aggregate_fit` method.
Here, indeed, is where the aggregation of the local models happens.

1. Let's write the function. Take as input the relevant parameters.

2. Focus on `server_round`, `results` and `failures`.

3. Aggregation of local models. Conversion of the aggregated model in `NDArrays`.

4. Compute pseudo gradient as the difference between the aggregated model and the previous global model.

5. Apply the server learning rate to the pseudo gradient.

6. Compute the norm of the pseudo gradient after having applied the server learning rate.

7. Sum the pseudo gradient to the previous global model.

8. Convert the updated global model to `Parameters`.

9. Replace `self.previous_parameters` with the updated global model.

10. Aggregate custom metrics if `fit_metrics_aggregation_fn` was provided.

11. Add the norm of the pseudo gradients to the metrics and print it.

12. Return the aggregated parameters and metrics.

```python

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average with learning rate."""

        # ==== NO  CHANGES HERE ====
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results in NDArrays
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # ==========================

        # Aggregate the results
        parameters_round = aggregate(weights_results)
        # Convert previous global model parameters to NDArrays
        parameters_start = parameters_to_ndarrays(self.previous_parameters)

        # Compute the pseudo gradients between the previous global model and the aggregated model
        # NOTE: remember that updates are the opposite of gradients
        pseudo_gradient: NDArrays = compute_model_delta(
            parameters_round, parameters_start
        )

        # Apply the learning rate to the pseudo gradients
        pseudo_gradient = [
            layer * self.server_learning_rate for layer in pseudo_gradient
        ]

        # Compute the norm of the pseudo gradients to be printed as a metric
        update_norm = compute_norm(pseudo_gradient)

        # Apply the pseudo gradients to the previous global model
        parameters_aggregated = [
            x + y for x, y in zip(parameters_start, pseudo_gradient)
        ]

        # Update current weights
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        self.previous_parameters = parameters_aggregated

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # Add the norm of the pseudo gradients to the metrics and print it
        metrics_aggregated["update_norm"] = update_norm
        log(INFO, f'FedAvgLearningRate :: aggregate_fit --- the norm of the update is {update_norm}')
        
        # Return the aggregated parameters and metrics
        return parameters_aggregated, metrics_aggregated
```

### Step 5: Modify the `server.py`

We also need to modify the `server.py` script to use our new strategy.
We want the script to take the server learning rate as an argument.
We also want it to use the `FedAvgLearningRate` strategy instead of the `FedAvg` strategy.

```python
from tutorial_1_centralised_to_federated.solution.server import main, fit_config
import argparse
from strategy import FedAvgLearningRate # <--- NEW


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

        strategy = FedAvgLearningRate( # <--- NEW
            min_available_clients=2,
            fraction_fit=1.0,
            min_fit_clients=2,
            fraction_evaluate=1.0,
            min_evaluate_clients=2,
            server_learning_rate=args.server_learning_rate, # <--- NEW
            # evaluate_fn=get_evaluate_fn(model, args.toy),
            on_fit_config_fn=fit_config,
            # on_evaluate_config_fn=evaluate_config,
        )

        main(args, strategy=strategy)


execute()
```

### Step 6: Modify the `run.sh` script

Before eventually running the experiment, we need the last modification to our files.
We modify `run.sh` to pass the server learning rate as an argument to the `server.py` script.

```bash
#!/bin/bash

# To make sure that relative paths work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

NUM_CLIENTS=2
ALPHA=100000

SERVER_LR=1.0 # <--- NEW

####
ALPHA_TEMP=$(printf %.2f $ALPHA)
PARTITIONS_DIR="dataset/lda/${NUM_CLIENTS}/${ALPHA_TEMP}/"
echo "Starting server"
python server.py --num_clients $NUM_CLIENTS --server_learning_rate $SERVER_LR  & 
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Create partitions"
python ../tutorial_1_centralised_to_federated/solution/create_partitions.py

for (( i=0; i<$NUM_CLIENTS; i++ ))
do
    echo "Starting client $i"
    python ../tutorial_1_centralised_to_federated/solution/client.py --cid $i --partitions_root $PARTITIONS_DIR &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
```

### Step 7: Run the experiment with different server learning rates

In the end, we can run the experiment.

```bash
./run.sh
```

Let's now modify the `run.sh` script to run the experiment with different server learning rates.

Change

```bash
SERVER_LR=1.0 # <--- NEW
```

to

```bash
SERVER_LR=0.5 # <--- NEW
```

### Step 8: Analyse the results

Show the server's outputs regarding the norm of the updates.

### Step 9: Conclusions

Second slide of the presentation.
