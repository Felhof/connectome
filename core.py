import abc
import itertools
import math
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import List, Callable, Union, Optional, Iterable

import graphviz
import plotly.express as px
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name

from utils import Connexion, coerce_int

# Types

Metric = Callable[[Float[Tensor, "seq vocab"]], float]
"""A metric is a function that take the logits on a prompt and returns how well the model is doing."""

# Interventions


class Intervention(abc.ABC):
    """An intervention is a collection of hooks that modify the behavior of a model."""

    filter_hook_name: Union[str, tuple[str, ...]] = ""

    def filter(self, name: str):
        """Return True if the hook should be applied to this activation.
        Refer to https://app.excalidraw.com/l/9KwMnW35Xt8/6PEWgOPSxXH for activation names.
        """
        assert self.filter_hook_name != "", "Must set filter_hook_name in subclass"
        return name.endswith(self.filter_hook_name)

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

    def batch_hook(
        self,
        activation: Float[Tensor, "batch *activation"],
        hook: HookPoint,
        sources: Iterable[Union[int, slice]],
        targets: Iterable[Union[int, slice]],
    ):
        for i, (source, target) in enumerate(zip(sources, targets)):
            self.hook(activation[i:i + 1], hook, source, target)

    @contextmanager
    def batch_hooks(
        self,
        model: HookedTransformer,
        sources: list[Union[int, slice]],
        targets: list[Union[int, slice]],
    ):
        assert len(sources) == len(targets)
        assert all(isinstance(source, (int, slice)) for source in sources)
        assert all(isinstance(target, (int, slice)) for target in targets)

        with model.hooks(fwd_hooks=[(
                self.filter,
                partial(self.batch_hook, sources=sources, targets=targets),
        )]):
            yield


class DampenIntervention(Intervention):
    filter_hook_name = 'pattern'

    def __init__(self, dampening_factor: float):
        self.dampening_factor = dampening_factor

    def hook(
        self,
        activation: Float[Tensor, "batch head seq_query seq_key"],
        hook: HookPoint,
        source: Union[int, slice],
        target: Union[int, slice],
    ):
        activation[:, :, target, source] *= self.dampening_factor


class ZeroPattern(DampenIntervention):

    def __init__(self):
        super().__init__(0.0)


class StopComputation(Exception):
    pass

class BaseCorruptIntervention(Intervention):
    filter_hook_name = 'z'

    def __init__(self, model: HookedTransformer, clean_input: str):
        if not model.cfg.use_split_qkv_input:
            warnings.warn("The model does not use split qkv input. Setting it to True.")
            model.cfg.use_split_qkv_input = True

        self.model = model
        self.clean_input = clean_input

        self.clean_cache = model.run_with_cache(
            clean_input, names_filter=lambda name: name.endswith("resid_pre"))[1]

    def corrupt_source(self, activation: Float[Tensor, "batch seq head d_head"],
                       hook: HookPoint, source: int, target: int):
        raise NotImplementedError

    def hook(
        self,
        main_activation: Float[Tensor, "batch seq head d_head"],
        hook: HookPoint,
        source: Union[int, slice],
        target: Union[int, slice],
    ):
        # This is not so great. We avoid recursively calling this function
        # removing all the hooks. And we don't add them back afterward.
        hook.remove_hooks()

        # Step 1: compute the attention score between clean query and corrupted key
        layer = hook.layer()

        def store_score(activation: Float[Tensor,
                                          "batch head seq_query seq_key"],
                        hook: HookPoint):
            hook.ctx["score"] = activation[:, :, target, source]
            raise StopComputation()

        with self.model.hooks(fwd_hooks=[
            (get_act_name("k", layer), partial(self.corrupt_source, source=source, target=target)),
            (get_act_name("attn_scores", layer), store_score),
        ]):
            clean_resid_pre = self.clean_cache[get_act_name(
                "resid_pre", layer)]
            try:
                self.model.blocks[layer](clean_resid_pre)
            except StopComputation:
                pass

        # Step 2: compute the new z

        def corrupt_score(activation: Float[Tensor,
                                            "batch head seq_query seq_key"],
                          hook: HookPoint):
            activation[:, :, target, source] = hook.ctx.pop("score")

        def hook_z(activation: Float[Tensor, "batch seq head d_head"],
                   hook: HookPoint):
            main_activation[:, target] = activation[:, target]
            raise StopComputation()

        with self.model.hooks(fwd_hooks=[
            (get_act_name("v", layer), partial(self.corrupt_source, source=source, target=target)),
            (get_act_name("attn_scores", layer), corrupt_score),
            (get_act_name("z", layer), hook_z),
        ]):
            try:
                self.model.blocks[layer](clean_resid_pre)
            except StopComputation:
                pass

class CorruptIntervention(BaseCorruptIntervention):
    def __init__(self, model: HookedTransformer, clean_input: str, corrupt_input: str):
        assert len(model.to_tokens(clean_input)) == len( model.to_tokens(corrupt_input))
        super().__init__(model, clean_input)
        self.corrupt_input = corrupt_input
        self.corrupt_cache = model.run_with_cache(
            corrupt_input,
            names_filter=lambda name: name.endswith(("k", "v")),
            remove_batch_dim=True,
        )[1]  # Ignore the logits

    def corrupt_source(self, activation: Float[Tensor, "batch seq head d_head"],
                       hook: HookPoint, source: int, target: int):
        activation[:, source] = self.corrupt_cache[hook.name][source]

class CropIntervention(BaseCorruptIntervention):

    def corrupt_source(self, activation: Float[Tensor, "batch seq head d_head"], hook: HookPoint, source: int,
                       target: int):
        assert isinstance(source, int), "source must be int for crop intervention"
        activation[:, source] = self.corrupt_caches[source][hook.name][min(source, 1)]

    def __init__(self, model: HookedTransformer, clean_input: str):
        super().__init__(model, clean_input)

        clean_tokens = model.to_tokens(clean_input)[0].tolist()
        bos_token = clean_tokens[0]
        self.corrupt_caches = [
            model.run_with_cache(
                torch.tensor([bos_token] +
                             clean_tokens[max(1, start):]),
                names_filter=lambda n: n.endswith(("v", 'k')),
                remove_batch_dim=True,
            )[1] for start in range(len(clean_tokens))
        ]

# Metrics


def logit_diff_metric(model: HookedTransformer, correct: str, *incorrect: str) -> Metric:
    correct_param_id = model.to_single_token(correct)
    incorrect_param_ids = [
        model.to_single_token(incorrect) for incorrect in incorrect
    ]

    def metric(
        logits: Float[Tensor, "seq vocab"],
    ) -> float:
        incorrect_logit = logits[-1, incorrect_param_ids].max()
        correct_logit = logits[-1, correct_param_id]
        return correct_logit - incorrect_logit

    return metric


# Exploration strategies

EndPoint = Union[int, slice]


class Strategy:
    """A strategy defines the which interventions to do, based on the results of previous interventions.
    It populates the `to_explore` set with pairs of endpoints to explore, and the `report` method is called
    when the results of an intervention are available.
    """

    def __init__(self):
        self.to_explore: list[tuple[EndPoint, EndPoint]] = []

    def start(self, n_tokens: int):
        raise NotImplementedError

    def pop_explorations(
        self,
        max_explorations: Optional[int] = None
    ) -> list[tuple[EndPoint, EndPoint]]:
        if max_explorations is None:
            max_explorations = len(self.to_explore)

        new = self.to_explore[:max_explorations]
        del self.to_explore[:max_explorations]
        return new

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.to_explore) == 0:
            raise StopIteration
        return self.to_explore.pop()

    def report(self, source: Union[int, slice], target: Union[int, slice],
               strength: float):
        pass


class BasicStrategy(Strategy):

    def start(self, n_tokens: int):
        self.to_explore = [(source, target) for target in range(1, n_tokens)
                           for source in range(1, target + 1)]

    def __len__(self):
        return len(self.to_explore)


class BisectStrategy(Strategy):

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def start(self, n_tokens: int):
        all_tokens = slice(0, n_tokens)
        self.to_explore = [(all_tokens, all_tokens)]

    def report(self, source: EndPoint, target: EndPoint, strength: float):
        assert type(source) == type(target) == slice

        # If the connexion is too weak, we can skip the whole square
        if abs(strength) < self.threshold:
            return

        if source.stop - source.start == 1:
            source_parts = [source]
        else:
            source_mid = math.ceil((source.start + source.stop) / 2)
            source_parts = [slice(source.start, source_mid),
                            slice(source_mid, source.stop)]

        if target.stop - target.start == 1:
            target_parts = [target]
        else:
            target_mid = math.ceil((target.start + target.stop) / 2)
            target_parts = [slice(target.start, target_mid),
                            slice(target_mid, target.stop)]

        # If we can't bisect anymore, because it's a single token
        if len(source_parts) == 1 and len(target_parts) == 1:
            return

        for source_part in source_parts:
            for target_part in target_parts:
                # If all sources are after all targets, we can skip the whole square
                if source_part.start < target_part.stop:
                    self.to_explore.append((source_part, target_part))

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

    def report(self, source: EndPoint, target: EndPoint, strength: float):
        assert type(source) == type(target) == int
        # If the connexion is too weak, don't explore it
        if abs(strength) >= self.threshold:
            self.to_explore.extend(self.new_to_visit(source))


class BacktrackBisectStrategy(Strategy):
    """A strategy that explores targets only when they are directly connected to the last token
    and bisects the sources to find possible earlier nodes."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.seen = set()

    def start(self, n_tokens: int):
        self.seen.clear()
        self.to_explore = [(slice(1, n_tokens), n_tokens - 1)]

    def report(self, source: EndPoint, target: EndPoint, strength: float):
        assert isinstance(target, int)
        assert isinstance(source, slice)

        if abs(strength) < self.threshold:
            return
        if source.stop == source.start + 1:
            # source.start is individually connected to target
            # -> We explore the possible parents of source.start
            if source.start not in self.seen:
                self.seen.add(source.start)
                self.to_explore.append((slice(1,
                                              source.start + 1), source.start))
        else:
            mid = math.ceil((source.start + source.stop) / 2)
            self.to_explore.append((slice(source.start, mid), target))
            self.to_explore.append((slice(mid, source.stop), target))


class SplitStrategy(Strategy):
    """A strategy that groups tokens into clusters and explores the connections between clusters
    before exploring the connections inside clusters. """

    DEFAULT_SPLITS = (
        '\n\n',
        # '\n',
        tuple('.!?'),
        tuple(',:;'),
    )

    def __init__(self, model: HookedTransformer, prompt: str, threshold: float,
                 delimiters: tuple[Union[str, tuple[str, ...]], ...] = DEFAULT_SPLITS):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.threshold = threshold
        # Make all delimiters tuples
        self.delimiters = tuple((delimiter,) if isinstance(delimiter, str) else delimiter
                                for delimiter in delimiters)
        # every delimiter should be a token
        for delimiters in self.delimiters:
            for delimiter in delimiters:
                tokens = self.model.to_str_tokens(delimiter, prepend_bos=False)
                assert len(tokens) == 1, f"Delimiter {delimiter} is not a single token but {tokens}."

        self.tree: dict[tuple[int, int], list[EndPoint]] = self.build_tree(model, prompt)

    def build_tree(self, model: HookedTransformer, prompt: str) -> dict[tuple[int, int], list[EndPoint]]:
        tokens = model.to_str_tokens(prompt)

        def new_child(parent: slice, child_start, child_end):
            # We include the delimiter in the previous slice
            child = slice(child_start, child_end + 1)
            # Always add the child to the current layer, to continue splitting it later
            current_layer.append(child)
            # Avoid loops
            if child != parent:
                tree[parent.start, parent.stop].append(child)

        tree: dict[tuple[int, int], list[EndPoint]] = defaultdict(list)
        last_layer = [slice(1, len(tokens))]
        # Depth by depth in the tree
        for delimiters in self.delimiters:
            current_layer = []
            for parent in last_layer:
                child_start = parent.start
                for child_end, token in enumerate(tokens[parent], start=parent.start):
                    # Check if the delimiter is (part of) the token (e.g. \n\n is part of \n\n\n)
                    if any(d in token for d in delimiters):
                        new_child(parent, child_start, child_end)
                        child_start = child_end + 1
                # We include the last slice, if it's not empty
                if child_start < parent.stop:
                    new_child(parent, child_start, parent.stop - 1)

            # Keep the parent slices since none have children of the kind
            if current_layer:
                last_layer = current_layer

        for parent in last_layer:
            # Slices of length one are already in the tree
            if parent.stop != parent.start + 1:
                tree[parent.start, parent.stop] = [
                    slice(t, t + 1) for t in range(parent.start, parent.stop)
                ]

        return tree

    def start(self, n_tokens: int):
        assert n_tokens == len(self.model.to_str_tokens(self.prompt))
        all_tokens = slice(1, n_tokens)
        self.to_explore = [(all_tokens, all_tokens)]

    def report(self, source: Union[int, slice], target: Union[int, slice],
               strength: float):
        if abs(strength) < self.threshold:
            return

        if isinstance(source, int) or source.stop == source.start + 1:
            starts = [source]
        else:
            starts = self.tree[source.start, source.stop]

        if isinstance(target, int) or target.stop == target.start + 1:
            ends = [target]
        else:
            ends = self.tree[target.start, target.stop]

        if len(ends) == 1 and len(starts) == 1:
            # We can't split the source or target further
            return

        for start in starts:
            for end in ends:
                end_stop = end.stop if isinstance(end, slice) else end
                start_start = start.start if isinstance(start, slice) else start
                if end_stop > start_start:
                    self.to_explore.append((start, end))

    def show_tree(self):
        tokens = self.model.to_str_tokens(self.prompt)
        positions = list(range(len(tokens)))
        for source, targets in sorted(self.tree.items()):
            print(f"{source}: {tokens[source[0]:source[1]]}")
            child_positions = []
            for target in targets:
                print(f"  -> {target}: {tokens[target]}")
                if isinstance(target, slice):
                    child_positions.extend(positions[target])
                else:
                    child_positions.append(positions[target])
            assert sorted(child_positions) == positions[source[0]:source[1]], (
                f"Child positions {child_positions} do not match parent positions "
                f"{positions[source[0]:source[1]]}"
            )



@torch.inference_mode()
def connectom(
    model: HookedTransformer,
    prompt: str,
    metric: Metric,
    intervention: Intervention,
    strategy: Strategy,
    max_batch_size: int = 10,
) -> list[Connexion]:
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    baseline_strength = metric(model(prompt)[0])
    print(f"Baseline strength: {baseline_strength:.2f}")

    connections: List[Connexion] = []
    strategy.start(n_tokens)
    try:
        total = len(strategy)
    except TypeError:
        total = None
    progress = tqdm(itertools.count(), desc="Exploring", unit=" connexions", total=total)
    while True:
        # Get the next source and target to explore
        next_explore = strategy.pop_explorations(max_batch_size)
        if not next_explore:
            break
        sources, targets = zip(*next_explore)
        progress.update(len(sources))
        # Show number of things in the queue
        progress.set_postfix({"to explore": len(strategy.to_explore)})

        # Evaluate it
        with intervention.batch_hooks(model, sources=sources, targets=targets):
            logits = model([prompt] * len(sources))

        for logit, source, target in zip(logits, sources, targets):
            strength = float(metric(logit))
            performance = (strength - baseline_strength) / baseline_strength

            strategy.report(source, target, performance)
            connections.append(
                Connexion(source, target, performance, "All layers"))

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
        coerce_int(endpoint)
        for connexion in connectome
        for endpoint in (connexion.source, connexion.target)
        if abs(connexion.strength) >= threshold
        and coerce_int(endpoint) is not None
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
                         connectome: List[Connexion], **plotly_kwargs):
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    # Maybe we want to sort the connexions by size of patch?
    connexions = torch.zeros((n_tokens, n_tokens))
    for connexion in sorted(connectome, key=lambda c: c.area, reverse=True):
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
        **plotly_kwargs,
    )


def map_range(value: float, min_value: float, max_value: float,
              min_range: float, max_range: float) -> float:
    """Map a value from a range to another"""
    normalized = (value - min_value) / (max_value - min_value)
    return min_range + normalized * (max_range - min_range)
