# %%
from functools import partial
from typing import Callable, List

import numpy as np
import torch
from transformer_lens import utils, HookedTransformer
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM

# %%
bert = BertForMaskedLM.from_pretrained("bert-large-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")


# %%
def to_str_tokens(tokenizer, prompt) -> List[str]:
    tokens = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
    return tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)


# TODO: make sure corrpted sentences have same number of tokens when using gpt2 tokenizer
def corrupt_sentences(model,
                      tokenizer,
                      sentences,
                      n_masks=4,
                      last_mask_idx=-1):
    model.eval()
    masked_sentences = []

    token_ids = tokenizer(sentences, return_tensors="pt",
                          padding=True)["input_ids"]

    n_batch = len(token_ids)
    s_len = len(token_ids[0])
    if last_mask_idx == -1:
        last_mask_idx = len(token_ids[0]) - 1
    assert last_mask_idx > n_masks, "last_mask_idx must be larger than n_masks."

    # mask tokens at n_mask random indices up to last_mask_idx
    # don't mask idx 0 or s_len as they contain the CLS and SEP tokens.
    # mask_indices = torch.randint(1, s_len, (n_batch, n_masks))
    mask_indices = (torch.sort(
        torch.randperm(last_mask_idx +
                       1)[:n_masks]).values.unsqueeze(0).expand(n_batch, -1) +
                    1)
    mask = torch.zeros(n_batch, s_len, dtype=torch.bool)
    mask.scatter_(1, mask_indices, True)
    token_ids.masked_fill_(mask, tokenizer.mask_token_id)

    masked_sentences = tokenizer.batch_decode(token_ids[:, 1:])

    # predict the masked tokens one by one
    for n, index_column in enumerate(mask_indices.T):
        out = model(token_ids).logits
        preds = out[token_ids == tokenizer.mask_token_id]
        top_preds = preds.max(-1).indices.reshape(n_batch, n_masks - n)
        token_ids[range(n_batch), index_column] = top_preds[range(n_batch), 0]

    corrupted_sentences = tokenizer.batch_decode(token_ids[:, 1:])

    return masked_sentences, corrupted_sentences


# %%

corrupt_sentences(
    bert,
    tokenizer,
    [
        # "The archaeologists discovered ancient ruins buried beneath the sand.",
        # "He took a sip of the warm tea to soothe his sore throat.",
        # "The hiker marveled at the breathtaking view from the mountaintop.",
        "Alice was so tired when she got back home, so she went to bed. She had a very nice dream.",
        "Lily likes cats and dogs. She asked her mum for a dog and her mum said no. So she asked her dad for a cat.",
        "Alice and Jack walked up the street and met a girl in a red dress. The girl smiled and said her name was Jane.",
    ],
    n_masks=8,
    last_mask_idx=16,
)


# %%
def patch_position(activation, hook, cache, source: int, target: int) -> None:
    activation[:, target, :, :] = cache[:, source, :, :]


# %%
gpt2_small = HookedTransformer.from_pretrained("gpt2-small")
# %%
original_prompt = "When Alice and Bob walked to the store, Bob gave a book to"
corrupted_prompt = "When Bob and Alice walked to the store, Alice gave a book to"
# %%
# Get cache from running clean and corrupted inputs
original_logits, cache = gpt2_small.run_with_cache(original_prompt)
_, corrupted_cache = gpt2_small.run_with_cache(corrupted_prompt)
# %%
original_logit_diff = (
    original_logits[0, -1, gpt2_small.to_single_token(" Alice")] -
    original_logits[0, -1, gpt2_small.to_single_token(" Bob")]).item()
# %%
#
# %%
# Create cache for patching by taking only the source position of the key and value vectors
# from the corrupted cache
source_pos = 10
target_pos = 14
for layer in range(gpt2_small.cfg.n_layers):
    cache[utils.get_act_name(
        "v", layer)][:, source_pos] = corrupted_cache[utils.get_act_name(
            "v", layer)][:, source_pos]
    cache[utils.get_act_name(
        "k", layer)][:, source_pos] = corrupted_cache[utils.get_act_name(
            "k", layer)][:, source_pos]
# %%
hooks = [(
    utils.get_act_name("v", layer),
    partial(
        patch_position,
        cache=cache[utils.get_act_name("v", layer)],
        source=source_pos,
        target=target_pos,
    ),
) for layer in range(gpt2_small.cfg.n_layers)] + [(
    utils.get_act_name("k", layer),
    partial(
        patch_position,
        cache=cache[utils.get_act_name("k", layer)],
        source=source_pos,
        target=target_pos,
    ),
) for layer in range(gpt2_small.cfg.n_layers)]
corrupted_logits = gpt2_small.run_with_hooks(original_prompt, fwd_hooks=hooks)
# %%
corrupted_logit_diff = (
    corrupted_logits[0, -1, gpt2_small.to_single_token(" Alice")] -
    corrupted_logits[0, -1, gpt2_small.to_single_token(" Bob")]).item()
# %%
