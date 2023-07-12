import abc
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Union, Iterable

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name

__all__ = [
    "Intervention",
    "DampenIntervention",
    "ZeroPattern",
    "BaseCorruptIntervention",
    "CorruptIntervention",
    "CropIntervention",
]


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

        with model.hooks(fwd_hooks=[(self.filter,
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

    def __repr__(self):
        return self.__class__.__name__ + "()"


class DampenIntervention(Intervention):
    filter_hook_name = "pattern"

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dampening_factor})"


class ZeroPattern(DampenIntervention):

    def __init__(self):
        super().__init__(0.0)


class StopComputation(Exception):
    pass


class BaseCorruptIntervention(Intervention):
    filter_hook_name = "z"

    def __init__(self, model: HookedTransformer, clean_input: str):
        if not model.cfg.use_split_qkv_input:
            warnings.warn("The model does not use split qkv input. Setting it to True.")
            model.cfg.use_split_qkv_input = True

        self.model = model
        self.clean_input = clean_input

        self.clean_cache = model.run_with_cache(
            clean_input, names_filter=lambda name: name.endswith("resid_pre"))[1]

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.clean_input})>"

    def corrupt_source(
            self,
            activation: Float[Tensor, "batch seq head d_head"],
            hook: HookPoint,
            source: int,
    ):
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

        def store_score(activation: Float[Tensor, "batch head seq_query seq_key"], hook: HookPoint):
            hook.ctx["score"] = activation[:, :, target, source]
            raise StopComputation()

        clean_resid_pre = self.clean_cache[get_act_name("resid_pre", layer)]
        with self.model.hooks(fwd_hooks=[
            (get_act_name("k", layer), partial(self.corrupt_source, source=source)),
            (get_act_name("attn_scores", layer), store_score),
        ]):
            try:
                self.model.blocks[layer](clean_resid_pre)
            except StopComputation:
                pass

        # Step 2: compute the new z

        def corrupt_score(activation: Float[Tensor, "batch head seq_query seq_key"],
                          hook: HookPoint):
            activation[:, :, target, source] = hook.ctx.pop("score")

        def hook_z(activation: Float[Tensor, "batch seq head d_head"], hook: HookPoint):
            main_activation[:, target] = activation[:, target]
            raise StopComputation()

        with self.model.hooks(fwd_hooks=[
            (get_act_name("v", layer), partial(self.corrupt_source, source=source)),
            (get_act_name("attn_scores", layer), corrupt_score),
            (get_act_name("z", layer), hook_z),
        ]):
            try:
                self.model.blocks[layer](clean_resid_pre)
            except StopComputation:
                pass


class CorruptIntervention(BaseCorruptIntervention):

    def __init__(self, model: HookedTransformer, clean_input: str, corrupt_input: str):
        assert len(model.to_tokens(clean_input)) == len(model.to_tokens(corrupt_input))
        super().__init__(model, clean_input)
        self.corrupt_input = corrupt_input
        self.corrupt_cache = model.run_with_cache(
            corrupt_input,
            names_filter=lambda name: name.endswith(("k", "v")),
            remove_batch_dim=True,
        )[1]  # Ignore the logits

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.clean_input} -> {self.corrupt_input})>"

    def corrupt_source(
            self,
            activation: Float[Tensor, "batch seq head d_head"],
            hook: HookPoint,
            source: int,
    ):
        activation[:, source] = self.corrupt_cache[hook.name][source]


class CropIntervention(BaseCorruptIntervention):

    def corrupt_source(
            self,
            activation: Float[Tensor, "batch seq head d_head"],
            hook: HookPoint,
            source: int,
    ):
        assert isinstance(source, int), "source must be int for crop intervention"
        activation[:, source] = self.corrupt_caches[source][hook.name][min(source, 1)]

    def __init__(self, model: HookedTransformer, clean_input: str):
        super().__init__(model, clean_input)

        clean_tokens = model.to_tokens(clean_input)[0].tolist()
        bos_token = clean_tokens[0]
        self.corrupt_caches = [
            model.run_with_cache(
                torch.tensor([bos_token] + clean_tokens[max(1, start):]),
                names_filter=lambda n: n.endswith(("v", "k")),
                remove_batch_dim=True,
            )[1] for start in range(len(clean_tokens))
        ]
