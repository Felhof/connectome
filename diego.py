import abc
import itertools
import math
from contextlib import contextmanager
from functools import partial
from typing import List, Callable, Union, Generator, Optional, Iterable

import circuitsvis.attention
import plotly.express as px
import graphviz
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils import Connexion

# Types

Metric = Callable[[Float[Tensor, "seq vocab"], Float[Tensor, "seq vocab"]],
float]
"""A metric is a function that take the original logits on a prompt and the patched logits 
after an intervention and returns a number."""



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
    def hook(
            self,
            activation: Float[Tensor, "batch *activation"],
            hook: HookPoint,
            source: Union[int, slice],
            target: Union[int, slice],
    ):
        """Modify the activation in-place."""
        raise NotImplementedError

    @contextmanager
    def hooks(
            self,
            model: HookedTransformer,
            source: Union[int, slice],
            target: Union[int, slice],
    ):
        assert isinstance(source, (int, slice))
        assert isinstance(target, (int, slice))

        with model.hooks(fwd_hooks=[(
                self.filter,
                partial(self.hook, source=source, target=target))]):
            yield


class ZeroPattern(Intervention):

    def filter(self, name: str):
        return name.endswith("pattern")

    def hook(
            self,
            activation: Float[Tensor, "batch head seq_query seq_key"],
            hook: HookPoint,
            source: Union[int, slice],
            target: Union[int, slice],
    ):
        activation[:, :, target, source] = 0.0


# Metrics


def logit_diff_metric(model: HookedTransformer, correct: str, incorrect: str):
    correct_token = model.to_single_token(correct)
    incorrect_token = model.to_single_token(incorrect)

    def metric(
            original_logits: Float[Tensor, "seq vocab"],
            patched_logits: Float[Tensor, "seq vocab"],
    ) -> float:
        original_diff = (original_logits[-1, correct_token] -
                         original_logits[-1, incorrect_token])
        patched_diff = (patched_logits[-1, correct_token] -
                        patched_logits[-1, incorrect_token])
        return (patched_diff - original_diff) / original_diff

    return metric


# Exploration strategies

EndPoint = Union[int, slice, list[int]]


class Strategy:
    """A strategy defines the which interventions to do, based on the results of previous interventions.
    It populates the `to_explore` set with pairs of endpoints to explore, and the `report` method is called
    when the results of an intervention are available.
    """

    def __init__(self):
        self.to_explore: list[tuple[EndPoint, EndPoint]] = []

    def start(self, n_tokens: int):
        raise NotImplementedError

    def pop_explorations(self, max_explorations: Optional[int] = None) -> list[tuple[EndPoint, EndPoint]]:
        if max_explorations is None:
            max_explorations = len(self.to_explore)

        new = self.to_explore[-max_explorations:]
        del self.to_explore[-max_explorations:]
        return new

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.to_explore) == 0:
            raise StopIteration
        return self.to_explore.pop()

    def report_all(self, source: EndPoint, target: EndPoint, strength: float):
        assert type(source) == type(target)
        if isinstance(source, list):
            for s, t in zip(source, target):
                self.report(s, t, strength)

    @abc.abstractmethod
    def report(self, source: Union[int, slice], target: Union[int, slice], strength: float):
        pass


class BasicStrategy(Strategy):
    def start(self, n_tokens: int):
        self.to_explore = [
            (source, target)
            for target in range(1, n_tokens)
            for source in range(1, target + 1)
        ]

    def __len__(self):
        return len(self.to_explore)


class BisectStrategy(Strategy):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def start(self, n_tokens: int):
        all_tokens = slice(0, n_tokens)
        self.to_explore = [(all_tokens, all_tokens)]

    def report(self, source: Union[int, slice], target: Union[int, slice], strength: float):
        assert type(source) == type(target) == slice

        # If the connexion is too weak, we can skip the whole square
        if abs(strength) < self.threshold:
            return

        if source.stop - source.start == 1:
            # We have an interval of length 1, so we can't split it any further
            assert target.stop - target.start == 1
        else:
            # Split the intervals in half and add the 4 new intervals to the set
            source_mid = math.ceil((source.start + source.stop) / 2)
            target_mid = math.ceil((target.start + target.stop) / 2)
            for start in [slice(source.start, source_mid), slice(source_mid, source.stop)]:
                for stop in [slice(target.start, target_mid), slice(target_mid, target.stop)]:
                    # If all sources are after all targets, we can skip the whole square
                    if start.start < stop.stop:
                        self.to_explore.append((start, stop))


class BacktrackingStrategy(Strategy):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.seen = set()

    def start(self, n_tokens: int):
        self.seen = set()
        self.to_explore = list(self.new_to_visit(n_tokens - 1))

    def new_to_visit(self, target: int):
        if target not in self.seen:
            self.seen.add(target)
            for source in range(1, target + 1):
                yield (source, target)

    def report(self, source: Union[int, slice], target: Union[int, slice], strength: float):
        assert type(source) == type(target) == int
        # If the connexion is too weak, don't explore it
        if abs(strength) >= self.threshold:
            self.to_explore.extend(self.new_to_visit(source))


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
    strategy.start(n_tokens)
    for source, target in tqdm(strategy):
        # Get the next source and target to explore
        # Evaluate it
        with intervention.hooks(model, source=source, target=target):
            logits = model(prompt)[0]
        strength = metric(original_predictions, logits)

        if isinstance(strength, torch.Tensor):
            strength = strength.item()

        strategy.report(source, target, strength)
        connections.append(Connexion(source, target, strength, "All layers"))

    return connections


# Other visualization


def plot_graphviz_connectome(
        model: HookedTransformer,
        prompt: str,
        connectome: List[Connexion],
        threshold: float = 0.0,
) -> graphviz.Digraph:
    tokens = model.to_str_tokens(prompt)
    graph = graphviz.Digraph()

    # Add all the used nodes to the graph with their corresponding string
    tokens_used = {
        endpoint
        for connexion in connectome
        for endpoint in (connexion.source, connexion.target)
        if abs(connexion.strength) >= threshold and connexion.is_single_pair
    }
    for i, token in enumerate(tokens):
        if i in tokens_used:
            graph.node(str(i), label=f"{i}: {token!r}")

    # Add all the important connexions to the graph
    min_strength = min(abs(connexion.strength) for connexion in connectome)
    max_strength = max(abs(connexion.strength) for connexion in connectome)
    for connexion in connectome:
        if abs(connexion.strength) >= threshold and connexion.is_single_pair:
            graph.edge(
                str(connexion.source_int),
                str(connexion.target_int),
                label=f"{connexion.strength:.2f}",
                color="#87D37C" if connexion.strength < 0 else "#E52B50",
                penwidth=str(
                    int(
                        map_range(abs(connexion.strength), min_strength,
                                  max_strength, 1, 7))),
            )

    return graph


def plot_attn_connectome(model: HookedTransformer, prompt: str,
                         connectome: List[Connexion]):
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    # Maybe we want to sort the connexions by size of patch?
    connexions = torch.zeros((n_tokens, n_tokens))
    for connexion in connectome:
        connexions[connexion.target, connexion.source] = connexion.strength

    triu = torch.triu(torch.ones(n_tokens, n_tokens, dtype=torch.bool),
                      diagonal=1)
    connexions.masked_fill_(triu, float("nan"))

    labels = [f"{i}: {token!r}" for i, token in enumerate(tokens)]
    return px.imshow(
        connexions,
        x=labels,
        y=labels,
        labels=dict(x="Source", y="Target", color="Strength"),
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Attention connectome",
    )


def map_range(value: float, min_value: float, max_value: float,
              min_range: float, max_range: float) -> float:
    """Map a value from a range to another"""
    normalized = (value - min_value) / (max_value - min_value)
    return min_range + normalized * (max_range - min_range)
