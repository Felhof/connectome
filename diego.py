import abc
import itertools
import math
from contextlib import contextmanager
from functools import partial
from typing import List, Callable, Union, Generator, Optional

import graphviz
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils import Connexion

# Types

Metric = Callable[[Float[Tensor, "seq vocab"], Float[Tensor, "seq vocab"]], float]
"""A metric is a function that take the original logits on a prompt and the patched logits 
after an intervention and returns a number."""

Strategy = Callable[[int], Generator[tuple[slice, slice], float, None]]
"""A strategy is a function that takes the number of tokens in the prompt and returns a generator
that:
- repeatedly yields a source and target slice to explore
- receives a strength of the connexion between the source and target (according to the metric)
"""


# Interventions

class Intervention(abc.ABC):
    """An intervention is a collection of hooks that modify the behavior of a model."""

    @abc.abstractmethod
    def filter(self, name: str):
        """Return True if the hook should be applied to this activation.
        Refer to https://app.excalidraw.com/l/9KwMnW35Xt8/6PEWgOPSxXH for activation names.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hook(self, activation: Float[Tensor, "batch *activation"], hook: HookPoint,
             source: Union[int, slice], target: Union[int, slice]):
        """Modify the activation in-place."""
        raise NotImplementedError

    @contextmanager
    def hooks(self, model: HookedTransformer, source: Union[int, slice], target: Union[int, slice]):
        assert isinstance(source, (int, slice))
        assert isinstance(target, (int, slice))

        with model.hooks(fwd_hooks=[
            (self.filter, partial(self.hook, source=source, target=target))
        ]):
            yield


class ZeroPattern(Intervention):
    def filter(self, name: str):
        return name.endswith("pattern")

    def hook(self, activation: Float[Tensor, "batch head seq_query seq_key"], hook: HookPoint,
             source: Union[int, slice], target: Union[int, slice]):
        activation[:, :, target, source] = 0.0


# Metrics

def logit_diff_metric(model: HookedTransformer, correct: str, incorrect: str):
    correct_token = model.to_single_token(correct)
    incorrect_token = model.to_single_token(incorrect)

    def metric(original_logits: Float[Tensor, "seq vocab"],
               patched_logits: Float[Tensor, "seq vocab"]) -> float:
        original_diff = original_logits[-1, correct_token] - original_logits[-1, incorrect_token]
        patched_diff = patched_logits[-1, correct_token] - patched_logits[-1, incorrect_token]
        return patched_diff - original_diff

    return metric


# Exploration strategies

def explore_bisect(n_tokens: int, threshold: float) -> Generator[tuple[slice, slice], float, None]:
    """Explore the connexion between all pairs of tokens in the prompt by cutting the interval in half
    at each step.
    Strength: can skip whole square of interventions
    Weakness: does not notice when two connexion compensate each other in the same square
    """

    to_explore = {
        ((0, n_tokens), (0, n_tokens))
    }
    while to_explore:
        sources, targets = to_explore.pop()
        if targets[1] < sources[0]:
            # This interval is has no attention anyway, so we can skip it
            continue

        # Evaluate the connexion between the two intervals
        strength = yield slice(*sources), slice(*targets)

        if abs(strength) > threshold:
            if sources[1] - sources[0] == 1:
                # We have an interval of length 1, so we can't split it any further
                assert targets[1] - targets[0] == 1
            else:
                # Split the intervals in half and add the 4 new intervals to the set
                source_mid = math.ceil((sources[0] + sources[1]) / 2)
                target_mid = math.ceil((targets[0] + targets[1]) / 2)
                to_explore.update([
                    ((sources[0], source_mid), (targets[0], target_mid)),
                    ((sources[0], source_mid), (target_mid, targets[1])),
                    ((source_mid, sources[1]), (targets[0], target_mid)),
                    ((source_mid, sources[1]), (target_mid, targets[1])),
                ])


def explore_all_pairs(n_tokens: int):
    """Default exploration strategy: explore all pairs of tokens in the prompt."""
    for target in range(1, n_tokens):
        for source in range(1, target + 1):
            yield source, target


def explore_backtrack(n_tokens: int, threshold: float = 0.5):
    """Explore strong connexions starting from the last token and recursively find the important tokens.
    Strength: can skip analysis of whole destination tokens
    Weakness: works only with metrics that depend only on the last token and fails to discover
        part of the useful graph that are connected through multiple weak connexions"""
    useful = {n_tokens - 1}
    seen = set()
    while useful:
        target = useful.pop()
        if target in seen:
            continue
        seen.add(target)

        for source in range(1, target + 1):
            strength = yield source, target
            if abs(strength) > threshold and source not in seen:
                useful.add(source)

    return None


@torch.inference_mode()
def connectom(
        model: HookedTransformer,
        prompt: str,
        metric: Metric,
        intervention: Intervention,
        strategy: Strategy,
) -> list[Connexion]:
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    original_predictions = model(prompt)[0]

    connections: List[Connexion] = []
    explorer = strategy(n_tokens)
    strength = None
    for _ in tqdm(itertools.count()):
        # Get the next source and target to explore
        try:
            source, target = explorer.send(strength)
        except StopIteration:
            break
        # Evaluate it
        with intervention.hooks(model, source=source, target=target):
            logits = model(prompt)[0]
        strength = metric(original_predictions, logits)

        # We record only point-to-point connexions, for now
        if (s := coerce_int(source)) is not None and (t := coerce_int(target)) is not None:
            connections.append(Connexion(s, t, strength, "All layers"))

    return connections


def coerce_int(value: Union[int, slice]) -> Optional[int]:
    if isinstance(value, slice):
        if value.stop == value.start + 1:
            return value.start
        return None
    else:
        return value

# Other visualization

def plot_graphviz_connectome(
        model: HookedTransformer,
        prompt: str,
        connectome: List[Connexion],
        threshold: float = 0.0,
) -> graphviz.Digraph:
    tokens = model.to_str_tokens(prompt)
    token_labels = [f"{token.replace(' ', '␣')}[{i}]" for i, token in enumerate(tokens)]

    max_strength = max(abs(connexion.strength) for connexion in connectome)

    graph = graphviz.Digraph()
    for connexion in connectome:
        if abs(connexion.strength) < threshold:
            continue
        graph.edge(token_labels[connexion.source],
                   token_labels[connexion.target],
                   label=f"{connexion.strength:.2f}",
                   color="#87D37C" if connexion.strength < 0 else "#E52B50",
                     penwidth=str(int(1 + 5 * abs(connexion.strength) / max_strength)))

    return graph
