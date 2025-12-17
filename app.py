import streamlit as st
import torch
import joblib
import json
from transformers import BertTokenizer, BertModel

# --------------------
# Page setup
# --------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis App")
st.write("BERT embeddings + Random Forest classifier")

# --------------------
# Load models (cached)
# --------------------
@st.cache_resource
def load_all():
    device = torch.device("cpu")

    # Load trained Random Forest
    rf_model = joblib.load("sentiment_rf.pkl")

    # Load tokenizer from local files
    tokenizer = BertTokenizer(
        vocab_file="vocab.txt",
        tokenizer_config_file="tokenizer_config.json",
        special_tokens_map_file="special_tokens_map.json"
    )

    # Load BERT backbone from HuggingFace (NOT local)
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.to(device)
    bert.eval()

    # Load label map
    with open("label_map.json") as f:
        label_map = json.load(f)

    return rf_model, tokenizer, bert, label_map, device


rf_model, tokenizer, bert_model, label_map, device = load_all()

# --------------------
# Prediction function
# --------------------
def predict_sentiment(text: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        emb = bert_model(**enc).last_hidden_state.mean(1)

    emb = emb.cpu().numpy()
    pred = rf_model.predict(emb)[0]

    return label_map[str(pred)]

# --------------------
# UI
# --------------------
user_text = st.text_area(
    "Enter text:",
    height=150,
    placeholder="Type a review or tweet here..."
)

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(user_text)

        if result == "positive":
            st.success(f"✅ Sentiment: {result}")
        elif result == "negative":
            st.error(f"❌ Sentiment: {result}")
        else:
            st.info(f"⚖️ Sentiment: {result}")
