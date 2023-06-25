from functools import partial

import streamlit as st
from transformer_lens import HookedTransformer

import diego as d

st.title("Connectome Visualizer")

st.markdown("""
- **Intervention**: Setting the attention pattern to zero between a pair of tokens, for every layer and head.
- **Metric**: Proportion of logit difference impacted by the intervention.
- **Strategy**: Explore all pairs of tokens.
""")


@st.cache_resource
def load_model(name: str):
    return HookedTransformer.from_pretrained(name)


model_name = st.radio("Model", ["gpt2", "attn-only-4l"], horizontal=True)
model = load_model(model_name)

prompt = st.text_input(
    "Prompt", "When Mary and John went to the store, John gave a drink to")


def select_token(label: str, default: str) -> str:
    token = st.text_input(label, default)
    # Feedback
    try:
        token_id = model.to_single_token(token)
    except AssertionError:
        st.error(f"Token not in vocabulary: `{token!r}`")
        st.stop()
    else:
        st.write(f"Understood as: `{token!r}` ({token_id})")
    return token


col1, col2 = st.columns(2)
with col1:
    correct_token = select_token("Correct Token", " Mary")
with col2:
    incorrect_token = select_token("Incorrect Token", " John")


@st.cache_data
def get_connectome(prompt, correct_token, incorrect_token, model_name):
    model = load_model(model_name)  # To invalidate cache if model changes
    return d.connectom(
        model,
        prompt,
        d.logit_diff_metric(model, correct_token, incorrect_token),
        d.ZeroPattern(),
        d.BasicStrategy(),
    )


connectome = get_connectome(prompt, correct_token, incorrect_token, model_name)

tab_graphviz, tab_attention = st.tabs(["Graphviz", "Attention"])

with tab_graphviz:
    threshold = st.slider("Threshold", 0.1, 1.0, 0.2)
    graph = d.plot_graphviz_connectome(model,
                                       prompt,
                                       connectome,
                                       threshold=threshold)

    st.graphviz_chart(graph)

with tab_attention:
    st.plotly_chart(d.plot_attn_connectome(model, prompt, connectome))

st.write(model.to_str_tokens(prompt))
