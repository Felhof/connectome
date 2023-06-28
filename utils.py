# %%
from dataclasses import dataclass
from functools import partial
from itertools import chain, combinations
from typing import Callable, List, Optional, Tuple, Any, Union

from circuitsvis.attention import attention_pattern
from jaxtyping import Float
import plotly.graph_objects as go
import torch
from torch import Tensor
from tqdm.autonotebook import trange, tqdm
from transformer_lens import utils, HookedTransformer


# %%
def block_attention(activation, hook, source: int, target: int) -> None:
    activation[:, :, target, source] = float("-inf")


def block_attention_for_head(activation, hook, head: int, source: int,
                             target: int) -> None:
    activation[:, head, target, source] = float("-inf")


def block_score(activation, hook, source: int, target: int) -> None:
    activation[:, :, target, source] = 0


# %%
@torch.inference_mode()
def connectom(
    model: HookedTransformer,
    prompt: str,
    metric: Callable[[Float[Tensor, "seq vocab"], Float[Tensor, "seq vocab"]],
                     float],
    show=False,
) -> Tensor:
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)

    original_predictions = model(prompt)[0]

    connections = torch.full((n_tokens, n_tokens), float("nan"))
    for target in trange(1, n_tokens):
        for source in range(1, target + 1):
            logits = model.run_with_hooks(
                prompt,
                fwd_hooks=[
                    (
                        # lambda name: name.endswith("attn_scores"),
                        # partial(block_attention, source=source, target=target),
                        lambda name: name.endswith("pattern"),
                        partial(block_score, source=source, target=target),
                        # partial(patch_in_avg_attn, source=source, target=target),
                    ),
                    # (
                    #     lambda name: name.endswith("hook_v"),
                    #     partial(patch_in_avg_v, target=target)
                    # )
                ],
            )[0]
            c = metric(original_predictions, logits)
            connections[target, source] = c

    return connections


# %%
### TOOL B


def powerset(iterable, min_size: int = 1, max_size: Optional[int] = None):
    """
    Yield all the subsets of the iterable of size between min_size and max_size.

    Example:
        >>> powerset([1,2,3], 1, 2) -> (1,) (2,) (3,) (1,2) (1,3) (2,3)"""

    if max_size is None:
        max_size = len(iterable)

    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(min_size, max_size + 1))


@torch.inference_mode()
def layer_level_connectom(
    model: HookedTransformer,
    prompt: str,
    metric: Callable[[Float[Tensor, "seq vocab"], Float[Tensor, "seq vocab"]],
                     float],
    threshold: float = 0.55,
) -> List:
    original_predictions = model(prompt)[0]
    str_tokens = model.to_str_tokens(prompt)
    connections = connectom(model, prompt, metric, show=False)
    important_connections = (torch.abs(connections)
                             > threshold).nonzero().tolist()
    layer_powerset = list(
        powerset(range(model.cfg.n_layers), min_size=1, max_size=4))

    results = []

    for target, source in tqdm(important_connections):
        done = []
        for layers in tqdm(layer_powerset,
                           leave=False,
                           desc=f"{source} -> {target}"):
            if any(d <= set(layers) for d in done):
                continue
            logits = model.run_with_hooks(
                prompt,
                fwd_hooks=[
                    (
                        # utils.get_act_name("attn_scores", layer),
                        # partial(block_attention, source=source, target=target),
                        utils.get_act_name("pattern", layer),
                        partial(block_score, source=source, target=target),
                        # partial(patch_in_avg_attn, source=source, target=target),
                    ) for layer in layers
                ]
                # ] + [
                #     (
                #         utils.get_act_name("v", layer),
                #         # partial(block_attention, source=source, target=target),
                #         partial(patch_in_avg_v, target=target),
                #     ) for layer in layers
                # ],
            )[0]

            result = metric(original_predictions, logits)
            if abs(result) > threshold:
                print(
                    f"{source}, {str_tokens[source]} -> {target}, {str_tokens[target]} {layers} {result}"
                )
                results.append((source, target, layers, result))
                done.append(set(layers))

    return results


# %% Metrics


def kl_on_last_token(
    original_logits: Float[Tensor, "seq vocab"],
    patched_logits: Float[Tensor, "seq vocab"],
) -> float:
    assert original_logits.ndim == 2
    return torch.nn.functional.kl_div(
        patched_logits[-1].log_softmax(-1),
        original_logits[-1].log_softmax(-1),
        log_target=True,
        reduction="sum",
    ).item()


def ioi_metric(s_token_idx: int, io_token_idx: int) -> Callable[..., float]:

    def metric(
        original_logits: Float[Tensor, "seq vocab"],
        patched_logits: Float[Tensor, "seq vocab"],
    ) -> float:
        baseline = original_logits[-1,
                                   io_token_idx] - original_logits[-1,
                                                                   s_token_idx]
        logit_diff = patched_logits[-1,
                                    io_token_idx] - patched_logits[-1,
                                                                   s_token_idx]
        return (logit_diff - baseline).item()

    return metric


def docstring_metric(correct_param_id: int,
                     incorrect_param_ids: List[int]) -> Callable[..., float]:

    def metric(
        original_logits: Float[Tensor, "seq vocab"],
        patched_logits: Float[Tensor, "seq vocab"],
    ) -> float:
        original_incorrect_logit = max(
            original_logits[-1, incorrect_param_id].item()
            for incorrect_param_id in incorrect_param_ids)
        baseline = original_logits[-1,
                                   correct_param_id] - original_incorrect_logit
        patched_incorrect_logit = max(
            patched_logits[-1, incorrect_param_id].item()
            for incorrect_param_id in incorrect_param_ids)
        logit_diff = patched_logits[-1,
                                    correct_param_id] - patched_incorrect_logit
        return (logit_diff - baseline).item()

    return metric


def coerce_int(value: Union[int, slice]) -> Optional[int]:
    if isinstance(value, int):
        return value
    elif value.stop == value.start + 1:
        return value.start
    else:
        return None


def endpoint_to_start_end(endpoint: Union[int, slice]) -> Tuple[int, int]:
    if isinstance(endpoint, int):
        return endpoint, endpoint + 1
    else:
        return endpoint.start, endpoint.stop

@dataclass
class Connexion:
    source: Union[int, slice]
    target: Union[int, slice]
    strength: float
    note: Any

    @property
    def is_single_pair(self) -> bool:
        return self.area == 1

    @property
    def source_int(self) -> int:
        source = coerce_int(self.source)
        if source is None:
            raise ValueError(f"Cannot get single int from {self.source}")
        return source

    @property
    def target_int(self) -> int:
        target = coerce_int(self.target)
        if target is None:
            raise ValueError(f"Cannot get single int from {self.target}")
        return target

    @property
    def area(self) -> int:
        source_size = 1 if isinstance(self.source, int) else self.source.stop - self.source.start
        target_size = 1 if isinstance(self.target, int) else self.target.stop - self.target.start
        return source_size * target_size

    @property
    def source_tuple(self) -> Tuple[int, int]:
        return endpoint_to_start_end(self.source)

    @property
    def target_tuple(self) -> Tuple[int, int]:
        return endpoint_to_start_end(self.target)

    def is_subset(self, other: "Connexion"):
        source_start, source_end = self.source_tuple
        target_start, target_end = self.target_tuple
        other_source_start, other_source_end = other.source_tuple
        other_target_start, other_target_end = other.target_tuple
        return (
            other_source_start <= source_start <= source_end <= other_source_end
            and other_target_start <= target_start <= target_end <= other_target_end
        )

    def __hash__(self):
        source = self.source if isinstance(self.source, int) else (self.source.start, self.source.stop)
        target = self.target if isinstance(self.target, int) else (self.target.start, self.target.stop)
        return hash((source, target, self.strength, self.note))

    def __repr__(self) -> str:
        source = self.source if isinstance(self.source, int) else f"{self.source.start}:{self.source.stop}"
        target = self.target if isinstance(self.target, int) else f"{self.target.start}:{self.target.stop}"
        return f"<Connexion({source} -> {target}: {self.strength:.2f})>"

def sankey_diagram_of_connectome(
    model: HookedTransformer,
    prompt: str,
    connectome: List[Connexion],
    threshold: float = 0.0,
    show: bool = True,
):
    node_labels = [
        f"{idx}: {token!r}"
        for idx, token in enumerate(model.to_str_tokens(prompt)[1:], start=1)
    ]

    link_colors = []
    sources = []
    targets = []
    values = []
    link_labels = []

    max_connection_strength = max(
        abs(connexion.strength) for connexion in connectome)

    for connexion in connectome:
        if abs(connexion.strength) < threshold:
            continue
        sources.append(connexion.source - 1)
        targets.append(connexion.target - 1)
        opacity = abs(connexion.strength) / max_connection_strength * 0.8

        color = (f"rgba(255,0,0, {opacity})"
                 if connexion.strength > 0 else f"rgba(0,0,255, {opacity})")
        link_colors.append(color)
        values.append(abs(connexion.strength))
        link_labels.append(str(connexion.note))

    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
            ),
            link=dict(
                source=sources,  # indices correspond to labels
                target=targets,
                value=values,
                label=link_labels,
                color=link_colors,
            ),
        )
    ])

    fig.update_layout(title_text="Effect of Attention Knockout", font_size=10)
    if show:
        fig.show()
    return fig
