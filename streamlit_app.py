from functools import partial

import streamlit as st
from transformer_lens import HookedTransformer

import diego as d
from utils import sankey_diagram_of_connectome

st.title("Connectome Visualizer")


@st.cache_resource
def load_model(name: str):
    return HookedTransformer.from_pretrained(name)


model_name = st.selectbox("Model", ["gpt2", "attn-only-4l"])
model = load_model(model_name)

prompt = st.text_input("Prompt", "When Mary and John went to the store, John gave a drink to")

def select_token(label: str, default: str) -> str:
    token = st.text_input(label, default)
    try:
        token_id = model.to_single_token(token)
    except AssertionError:
        st.write(f"Token not in vocabulary: `{token!r}`")
        st.stop()
    else:
        st.write(f"Understood as: `{token!r}` ({token_id})")
    return token

col1, col2 = st.columns(2)
with col1:
    correct_token = select_token("Correct Token", " Mary")
with col2:
    incorrect_token = select_token("Incorrect Token", " John")

# Feedback


@st.cache_data
def get_connectome(prompt, correct_token, incorrect_token, model_name):
    model = load_model(model_name)  # To invalidate cache if model changes
    return d.connectom(model, prompt,
                d.logit_diff_metric(model, correct_token, incorrect_token),
                d.ZeroPattern(),
                strategy=d.explore_all_pairs,
                )

connectome = get_connectome(prompt, correct_token, incorrect_token, model_name)

threshold = st.slider("Threshold", 0.1, 1.0, 0.6)

# st.plotly_chart(
#     sankey_diagram_of_connectome(model, prompt, connectome, threshold=threshold, show=False)
# )
st.graphviz_chart(
    d.plot_graphviz_connectome(model, prompt, connectome, threshold=threshold)
)

st.markdown('''
- **Intervention**: Setting the attention score to zero at every layer between pairs of tokens.
- **Metric**: Logit difference between the correct token and the incorrect token, to which we remove the baseline from the original prompt.
- **Strategy**: Explore all pairs of tokens.
''')

st.write(model.to_str_tokens(prompt))