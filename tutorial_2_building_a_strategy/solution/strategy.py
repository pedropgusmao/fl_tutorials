from logging import INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import NDArrays, MetricsAggregationFn, FitRes, FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from shared.utils import compute_model_delta, compute_norm


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
        on_evaluate_config_fn: Optional[Callable[[
            int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0,  # <--- NEW
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
        self.server_learning_rate = server_learning_rate  # <--- NEW

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.previous_parameters = parameters  # <--- NEW
        return super().configure_fit(server_round, parameters, client_manager)

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

        # Aggregate the results
        parameters_round = aggregate(weights_results)
        
    
        # Convert previous global model parameters to NDArrays
        parameters_start = parameters_to_ndarrays(self.previous_parameters)
        
        # ==========================

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

        # ==== NO  CHANGES HERE ====
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
        # ==========================
        
        # Add the norm of the pseudo gradients to the metrics and print it
        metrics_aggregated["update_norm"] = update_norm
        log(INFO,
            f'FedAvgLearningRate :: aggregate_fit --- the norm of the update is {update_norm}')

        # Return the aggregated parameters and metrics
        return parameters_aggregated, metrics_aggregated
