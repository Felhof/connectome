"""Metrics for evaluating the performance of a model on a prompt."""

from typing import Callable

from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

Metric = Callable[[Float[Tensor, "seq vocab"]], float]
"""
A metric is a function that take the logits on a prompt and returns how well the model is doing.

The higher the value, the better the model is doing. 
A model that cannot to the task should return 0, and negative values means that the model does the opposite of the task.
"""


def logit_diff_metric(model: HookedTransformer, correct: str, *incorrect: str) -> Metric:
    """Get a metric that returns the logit difference between the correct token and the max of the incorrect tokens.

    Returns:
        f: (logits |-> logits[-1, correct] - max(logits[-1, incorrect]))
    """
    correct_param_id = model.to_single_token(correct)
    incorrect_param_ids = [model.to_single_token(incorrect) for incorrect in incorrect]

    def metric(logits: Float[Tensor, "seq vocab"]) -> float:
        assert logits.ndim == 2
        incorrect_logit = logits[-1, incorrect_param_ids].max()
        correct_logit = logits[-1, correct_param_id]
        return float(correct_logit - incorrect_logit)

    return metric


def correct_token_metric(model: HookedTransformer, correct: str) -> Metric:
    """Get a metric that returns the logit difference between the correct token and the other most likely token."""
    correct_token_id = model.to_single_token(correct)

    def metric(logits: Float[Tensor, "seq vocab"]) -> float:
        assert logits.ndim == 2
        correct_token_logit = logits[-1, correct_token_id]
        top = logits[-1].topk(2)
        if top.indices[0] == correct_token_id:
            alt_logit = top.values[1]
        else:
            alt_logit = top.values[0]
        return correct_token_logit - alt_logit

    return metric
