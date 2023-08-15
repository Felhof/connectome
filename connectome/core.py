import itertools
from typing import List, Union, Optional

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils import Connexion
from .interventions import Intervention
from .metrics import Metric
from .strategies import Strategy

__all__ = [
    "connectome",
    "filter_connectome",
    "cut_connectome",
]


# Types


@torch.inference_mode()
def connectome(
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
    assert baseline_strength > 0.5, "The model cannot do the task."

    connections: List[Connexion] = []
    strategy.start(n_tokens)
    try:
        # noinspection PyTypeChecker
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
            connections.append(Connexion(source, target, performance, "All layers"))

    return connections


# Other visualization
def filter_connectome(
        connectome: List[Connexion],
        depth: Optional[int] = None,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
) -> List[Connexion]:
    """Filter the connectome to only keep the connexion at a given depth.

    If depth is None, keep only the token-to-token connexions.
    Otherwise, it builds a tree based on subset ordering of the sources and targets
    and keeps only the connexions at the given depth and the leaves above it.
    """

    if depth is None:
        kept = [
            connexion for connexion in connectome
            if abs(connexion.strength) >= threshold and connexion.is_single_pair
        ]
        kept = sorted(kept, key=lambda c: abs(c.strength), reverse=True)[:top_k]
        return kept

    # Sort the connections, biggest area first
    connectome = sorted(connectome, key=lambda c: c.area, reverse=True)

    # Build the tree
    tree: dict[Union[None, Connexion], list[Connexion]] = {None: []}
    for connexion in connectome:
        # find the parent of the connexion
        possible_parents = []
        for parent_connexion in tree:
            if parent_connexion is not None and connexion.is_subset(parent_connexion):
                possible_parents.append(parent_connexion)
        parent = min(possible_parents, key=lambda c: c.area, default=None)
        tree[parent].append(connexion)
        tree[connexion] = []

    kept = []

    def recurse(connexion: Connexion, depth: int):
        if depth == 0:
            kept.append(connexion)
        elif not tree[connexion]:  # is a leaf
            kept.append(connexion)
        else:
            for child in tree[connexion]:
                recurse(child, depth - 1)

    for connexion in tree[None]:
        recurse(connexion, depth)

    # Filter out the ones that are too small
    kept = [c for c in kept if abs(c.strength) >= threshold]
    # Keep only the top k connections in strength
    kept = sorted(kept, key=lambda c: abs(c.strength), reverse=True)[:top_k]

    return kept


# Validation


@torch.inference_mode()
def cut_connectome(
        model: HookedTransformer,
        prompt: str,
        metric: Metric,
        connectome: list[Connexion],
        threshold: float = 0.0,
        keep_bos: bool = True,
        dampen_weak: float = 0.0,
):
    original_logits = model(prompt)[0]
    baseline = metric(original_logits)

    n_tokens = len(model.to_str_tokens(prompt))
    mask = torch.zeros((n_tokens, n_tokens)) + dampen_weak
    if keep_bos:
        mask[:, 0] = 1
    for connexion in connectome:
        if abs(connexion.strength) >= threshold:
            mask[connexion.target, connexion.source] = 1

    def hook(activation: Float[Tensor, "batch=1 n_head seq_query seq_key"], hook: HookPoint):
        activation *= mask

    logits = model.run_with_hooks(prompt, fwd_hooks=[(lambda name: name.endswith("pattern"), hook)])

    cut_value = metric(logits[0])
    strength_kept = float(cut_value / baseline)

    print("Model:", model.cfg.model_name)
    print("Prompt:", prompt)
    print(f"Cutting everything but {int(mask.sum())} connections")
    print(f"Baseline: {baseline:.3f}")
    print(f"Cut value: {cut_value:.3f}")
    print(f"Strength kept: {strength_kept:.3f}")

    return strength_kept
