from typing import List, Optional

import IPython.display
import graphviz
import plotly.express as px
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from connectome import filter_connectome
from utils import Connexion

__all__ = [
    "graphviz_connectome",
    "plot_graphviz_connectome",
    "attn_connectome",
    "plot_attn_connectome",
]


def graphviz_connectome(
        model: HookedTransformer,
        prompt: str,
        connectome: List[Connexion],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
) -> graphviz.Digraph:
    tokens = model.to_str_tokens(prompt)
    graph = graphviz.Digraph()

    # Keep only the connexions we care about
    connectome = filter_connectome(connectome, depth=depth, threshold=threshold, top_k=top_k)

    # Add all the used nodes to the graph with their corresponding string
    nodes = {
        endpoint
        for connexion in connectome
        for endpoint in (connexion.source_tuple, connexion.target_tuple)
    }
    for endpoint in nodes:
        if endpoint[0] + 1 == endpoint[1]:  # single token
            pos = f"{endpoint[0]}"
        else:
            pos = f"{endpoint[0]}:{endpoint[1]}"

        text = "".join(tokens[endpoint[0]:endpoint[1]])
        text = repr(text)
        # Shorten the text if it's too long
        if len(text) > 30:
            text = text[:13] + "..." + text[-13:]
        text = text.replace("\\", "\\\\")
        graph.node(str(endpoint), label=f"{pos}: {text}")

    # Add all the important connexions to the graph
    min_strength = min(abs(connexion.strength) for connexion in connectome)
    max_strength = max(abs(connexion.strength) for connexion in connectome)
    if min_strength == max_strength:
        min_strength = 0
    for connexion in connectome:
        graph.edge(
            str(connexion.source_tuple),
            str(connexion.target_tuple),
            label=f"{connexion.strength:.2f}",
            color="#87D37C" if connexion.strength < 0 else "#E52B50",
            penwidth=str(int(map_range(abs(connexion.strength), min_strength, max_strength, 1, 7))),
        )

    return graph


def plot_graphviz_connectome(
        model: HookedTransformer,
        prompt: str,
        connectome: List[Connexion],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        use_svg: bool = False,
):
    graph = graphviz_connectome(
        model,
        prompt,
        connectome,
        threshold=threshold,
        top_k=top_k,
        depth=depth,
    )
    if use_svg:
        out = IPython.display.SVG(graph.pipe(format="svg", encoding="utf8"))
    else:
        out = IPython.display.Image(graph.pipe(format="png"))

    IPython.display.display(out)
    return graph


def attn_connectome(
        model: HookedTransformer,
        prompt: str,
        connectome: List[Connexion],
        fill=float("nan"),
) -> Float[Tensor, "n_tokens n_tokens"]:
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    # Maybe we want to sort the connexions by size of patch?
    connexions = torch.zeros((n_tokens, n_tokens))
    for connexion in sorted(connectome, key=lambda c: c.area, reverse=True):
        connexions[connexion.target, connexion.source] = connexion.strength

    triu = torch.triu(torch.ones(n_tokens, n_tokens, dtype=torch.bool), diagonal=1)
    connexions.masked_fill_(triu, fill)
    return connexions


def plot_attn_connectome(model: HookedTransformer, prompt: str, connectome: List[Connexion],
                         **plotly_kwargs):
    connexions = attn_connectome(model, prompt, connectome)
    tokens = model.to_str_tokens(prompt)
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


def map_range(value: float, min_value: float, max_value: float, min_range: float,
              max_range: float) -> float:
    """Map a value from a range to another"""
    normalized = (value - min_value) / (max_value - min_value)
    return min_range + normalized * (max_range - min_range)
