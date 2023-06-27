# %%
import gc

from tqdm.autonotebook import tqdm
import numpy as np
import torch
from transformer_lens import HookedTransformer

import core

# %%
torch.set_grad_enabled(False)
# %%
model = HookedTransformer.from_pretrained("gpt2-small")
# %%
original_prompt = "When John and Mary went to the store, John gave a book to"


# %%
def get_similar_word(model, word, distance=1):

    def basically_equal(str1, str2):
        str1 = str1.replace(" ", "").lower()
        str2 = str2.replace(" ", "").lower()

        if str1 == str2:
            return True

        if str1 + "s" == str2 or str2 + "s" == str1:
            return True

        return False

    similarities = (
        model.W_E[model.to_single_token(word)] @ model.W_E.T).clone()
    similar_token_ids = torch.argsort(similarities, descending=True).tolist()
    found_similar_words = 0
    for token_id in similar_token_ids:
        similar_word = model.to_single_str_token(token_id)
        if basically_equal(word, similar_word):
            continue
        found_similar_words += 1
        if found_similar_words == distance:
            return similar_word
    return ""


# %%
get_similar_word(model, " Christmas", distance=7)
torch.cuda.empty_cache()
gc.collect()


# %%
def get_similar_word_softmax(model, word, temperature=1.0, k=20):
    similarities = (
        model.W_E[model.to_single_token(word)] @ model.W_E.T).clone()
    top_values, top_indices = torch.topk(similarities, k=k)
    probabilites = (top_values / temperature).softmax(0)
    replacement_id = top_indices[torch.distributions.Categorical(
        probabilites).sample()]
    return model.to_single_str_token(replacement_id.item())


# %%
def corrupt_prompt_with_similar_words(model, original_str_tokens,
                                      replace_pos_up_to, n_replacements,
                                      distance):
    str_tokens = original_str_tokens.copy()
    ids_to_replace = np.random.choice(range(1, replace_pos_up_to),
                                      n_replacements,
                                      replace=False)
    for id in ids_to_replace:
        str_tokens[id] = get_similar_word(model,
                                          str_tokens[id],
                                          distance=distance)
        torch.cuda.empty_cache()
        gc.collect()
    corrupted_prompt = model.tokenizer.batch_decode(
        [[model.to_single_token(t) for t in str_tokens][1:]])
    return corrupted_prompt


# %%
def corrupt_prompt_with_softmax(model, original_str_tokens, replace_pos_up_to,
                                temperature):
    str_tokens = original_str_tokens.copy()


# %%
distances = [4, 8, 12, 16]
n_samples = 20
n_replacements = 4
source_pos = [9, 10, 11]
replace_pos_up_to = min(source_pos)
target_pos = [14] * len(source_pos)
original_str_tokens = model.to_str_tokens(original_prompt)
metric = core.logit_diff_metric(model, " John", " Mary")
original_predictions = model(original_prompt)[0]

results = {pos: [] for pos in source_pos}
for distance in distances:
    corrupted_prompts = []
    distance_results = []
    for _ in range(n_samples):
        corrupted_prompt = corrupt_prompt_with_similar_words(
            model, original_str_tokens, replace_pos_up_to, n_replacements,
            distance)
        corrupted_prompts.append("".join(corrupted_prompt))

    for corrupt_prompt in tqdm(corrupted_prompts):
        intervention = core.CorruptIntervention(model, original_prompt,
                                                corrupt_prompt)  # type: ignore

        with intervention.batch_hooks(model,
                                      sources=source_pos,
                                      targets=target_pos):
            logits = model([original_prompt] * len(source_pos))

        for logit, source, target in zip(logits, source_pos, target_pos):
            strength = metric(original_predictions, logit)
            results[source].append(strength.item())

# %%
import plotly.express as px

x_axis = []
y_axis = []
colors = []
str_tokens = model.to_str_tokens(original_prompt)
color_map = {pos: str_tokens[pos] for pos in source_pos}

for pos in source_pos:
    y_axis.extend([abs(r) for r in results[pos]])
    for distance in distances:
        x_axis.extend([distance] * n_samples)
    colors.extend([color_map[pos]] * n_samples * len(distances))

fig = px.scatter(x=x_axis, y=y_axis, color=colors)

fig.show()

# %%
p = 90

y_axis = []
for pos in source_pos:
    values = []
    for idx, distance in enumerate(distances):
        pos_results_for_distance = [
            abs(r) for r in results[pos][idx * n_samples:(idx + 1) * n_samples]
        ]
        values.append(np.percentile(pos_results_for_distance, p))
    y_axis.append(values)
# %%
px.line(x=distances, y=y_axis, labels=[str_tokens[pos] for pos in source_pos])

# %%
