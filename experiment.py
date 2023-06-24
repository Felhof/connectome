# %%
from itertools import chain, combinations, product
from functools import partial
from sklearn import base
from typing import List

import numpy as np
import torch
from transformer_lens import HookedTransformer

from utils import (
    docstring_metric,
    ioi_metric,
    kl_on_last_token,
    layer_level_connectom,
    sankey_diagram_of_connectome 
)


# %%
model = HookedTransformer.from_pretrained("gpt2-small")
# %%
prompt = "When Mary and John went to the store, John gave a drink to"
# %%
s_token_id = model.to_single_token(" John")
io_token_id = model.to_single_token(" Mary")
results = layer_level_connectom(
    model,
    prompt, 
    ioi_metric(s_token_id, io_token_id), 
    threshold=0.75
)
# %%
results = layer_level_connectom(
    model,
    prompt, 
    kl_on_last_token, 
    threshold=0.05
)
# %%
sankey_diagram_of_connectome(model, prompt, results)
# %%
docstring_prompt1 = """"def old(self, first, files, page, names, size, read):
    \"\"\"sector gap population

    :param page: message tree
    :param names: detail mine
    :param """

correct_param1 = " size"
incorrect_params1 = [
    " self", " first", " files", " page", " names", " read"
]

docstring_prompt2 = """"def port(self, load, size, file, last):
    \"\"\"oil column piece

    :param load: crime population
    :param size: unit dark
    :param """

correct_param2 = " file"
incorrect_params2 = [
    " self", " load", " size", " last"
]
# %%
four_layer_attn_only = HookedTransformer.from_pretrained("attn-only-4l")
# %%
# %%
def get_model_completions(prompt):
    logits = four_layer_attn_only(prompt)
    _, indices = torch.topk(logits[0,-1], 10)
    print(four_layer_attn_only.to_str_tokens(indices))
# %%
docstring_results = layer_level_connectom(
    model,
    docstring_prompt1, 
    kl_on_last_token, 
    threshold=0.2
)

# %%
def map_connectome_for_docstring_task(
    model,
    prompt,
    correct_param: str,
    incorrect_param: List[str],
    threshold=1.
):
    correct_param_id = int(model.to_single_token(correct_param))
    incorrect_param_ids = [
        int(model.to_single_token(token)) 
        for token in incorrect_param

    ]
    docstring_results = layer_level_connectom(
        model,
        prompt, 
        docstring_metric(correct_param_id, incorrect_param_ids), 
        threshold=threshold
    )
    sankey_diagram_of_connectome(model, prompt, docstring_results)


# %%
map_connectome_for_docstring_task(
    model, 
    docstring_prompt1,
    correct_param1,
    incorrect_params1,
)
# %%
map_connectome_for_docstring_task(
    four_layer_attn_only, 
    docstring_prompt2,
    correct_param2,
    incorrect_params2,
    threshold=1.5
)

# %%
