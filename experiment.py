# %%
from functools import partial
from typing import Callable, Optional, Tuple
# from cv2 import log
from sklearn import base

import torch
from transformer_lens import HookedTransformer
from torch import Tensor
from tqdm.auto import trange
from circuitsvis.attention import attention_pattern
from IPython.display import display
import plotly.express as px
from jaxtyping import Int, Float

# %%
model = HookedTransformer.from_pretrained("gpt2-small")
# %%
prompts = []

names = ["Mark", "Tom", "Diego", "Jane", "Matt", "Max"]

for s1 in names:
    for s2 in names:
        for io in names:
            prompts.append(
                f"When {s1} and {io} went to the store, {s2} gave a drink to" 
            )
# %%
with torch.inference_mode():
    _, generic_cache = model.run_with_cache(prompts) 

# %%
def patch_in_avg(activation, hook, source: int, target: int) -> None:
    activation[:, :, target, source] = generic_cache[hook.name][:, :, target, source].mean(0)

# %%
def block_attention(activation, hook, source: int, target: int) -> None:
    activation[:, :, target, source] = float("-inf")

def block_score(activation, hook, source: int, target: int) -> None:
    activation[:, :, target, source] = 0

@torch.inference_mode()
def connectom(prompt: str, metric: Callable[[Float[Tensor, 'seq vocab'], Float[Tensor, 'seq vocab']], float]) -> None:
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)

    original_predictions = model(prompt)[0]

    connections = torch.full((n_tokens, n_tokens), float("nan"))
    for target in trange(1, n_tokens):
        for source in range(1, target+1):
            logits = model.run_with_hooks(
                prompt,
                fwd_hooks=[
                    (
                        lambda name: name.endswith("attn_scores"),
                        partial(block_attention, source=source, target=target),
                        # lambda name: name.endswith("pattern"),
                        # partial(block_score, source=source, target=target),
                        # partial(patch_in_avg, source=source, target=target),
                    )
                ],
            )[0]
            connections[target, source] = metric(original_predictions, logits)

    display(attention_pattern(tokens, connections))
    # return

    tokens_labels = [f"{i}: {t}" for i, t in enumerate(tokens)]
    px.imshow(connections,
        x=tokens_labels,
        y=tokens_labels,
        labels=dict(x="Source", y="Target"),
    ).show()

# %% Metrics

def kl_on_last_token(original_logits, patched_logits) -> float:
    return torch.nn.functional.kl_div(
        patched_logits[-1].log_softmax(-1),
        original_logits[-1].log_softmax(-1),
        log_target=True,
        reduction="sum",
    ).item()

def ioi_metric(subject: str, indirect_object: str) -> float:
    s = model.to_single_token(subject)
    io = model.to_single_token(indirect_object)
    def metric(original_logits: Float[Tensor, 'seq vocab'],
               patched_logits: Float[Tensor, 'seq vocab'],
              ) -> float:
        baseline = original_logits[-1, io] - original_logits[-1, s]
        logit_diff = patched_logits[-1, io] - patched_logits[-1, s]
        return logit_diff - baseline
    return metric

# %%

prompt = "When Diego and Felix went to ARENA, Felix gave two paperclips to"
prompt = "When Mary and John went to the store, John gave a drink to"
# %%
connectom(prompt, ioi_metric(" John", " Mary"))
# %%
connectom(prompt, kl_on_last_token)

# %%
connectom(" ( ( ) ( ) ) ( ) ( ( ( ) ( ) ) )")

# %%

