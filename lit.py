from functools import partial

import streamlit as st
from transformer_lens import HookedTransformer

import diego as d
from utils import sankey_diagram_of_connectome

st.title("Connectome Visualizer")


@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained("gpt2")


model = load_model()

prompt = st.text_input("Prompt", "When Mary and John went to the store, John gave a drink to")
correct_token = st.text_input("Correct Token", " Mary")
incorrect_token = st.text_input("Incorrect Token", " John")

# Feedback
st.write(f"Correct token: `{correct_token!r}`")
st.write(f"Incorrect token: `{incorrect_token!r}`")


@st.cache_data
def get_connectome(prompt, correct_token, incorrect_token):
    return d.connectom(model, prompt,
                d.logit_diff_metric(model, correct_token, incorrect_token),
                d.ZeroPattern(),
                strategy=d.explore_all_pairs,
                )

connectome = get_connectome(prompt, correct_token, incorrect_token)

threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

# st.plotly_chart(
#     sankey_diagram_of_connectome(model, prompt, connectome, threshold=threshold, show=False)
# )
st.graphviz_chart(
    d.plot_graphviz_connectome(model, prompt, connectome, threshold=threshold)
)

st.write(model.to_str_tokens(prompt))