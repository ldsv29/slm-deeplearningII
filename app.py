import torch
import streamlit as st
from transformers import AutoTokenizer
from config import SLMConfig, SLMModel

MODEL_ID = "ldsv29/slm-midtraining"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SLMModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer, device

def generate(model, tokenizer, device, history, max_new_tokens, temperature, top_k, top_p):
    prompt = ""
    for role, content in history:
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


st.title("SLM Chat")
st.caption(f"Modelo: {MODEL_ID}")

with st.sidebar:
    st.header("Parâmetros de geração")
    max_new_tokens = st.slider("Max new tokens", min_value=10, max_value=512, value=200, step=10)
    temperature    = st.slider("Temperature",     min_value=0.1, max_value=2.0, value=0.8, step=0.05)
    top_k          = st.slider("Top-k",           min_value=0,   max_value=200, value=50,  step=5)
    top_p          = st.slider("Top-p",           min_value=0.1, max_value=1.0, value=1.0, step=0.05)
    st.divider()
    if st.button("Limpar conversa"):
        st.session_state.history = []
        st.rerun()

model, tokenizer, device = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.write(content)

if user_input := st.chat_input("Digite sua mensagem..."):
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Gerando..."):
            response = generate(model, tokenizer, device, st.session_state.history,
                                max_new_tokens, temperature, top_k, top_p)
        st.write(response)

    st.session_state.history.append(("assistant", response))