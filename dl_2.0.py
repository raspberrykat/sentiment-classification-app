import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_sentiment_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        device_map=DEVICE # Memuat bobot langsung ke device target
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    return tokenizer, model

try:
    model_path = "./bert-rotten-output"
    tokenizer, model = load_sentiment_model(model_path)
    
    st.set_page_config(page_title="Sentiment Prediction for Movie Review 🍿")
    
except Exception as e:
    st.error(f"Failed to load the model or tokenizer. Error: {e}")
    st.stop()


def predict(text, tokenizer, model):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return "FRESH (POSITIVE) 🍅" if pred == 1 else "ROTTEN (NEGATIVE) 🤢"


# UI
st.title("🎬 Sentiment Classification for Movie Review using BERT")
st.subheader("Model was trained with the Rotten Tomatoes Review Dataset")

user_input = st.text_area(
    "Input your review here:",
    "This movie is amazing, i'll give it a 5 out of 5. It's very inspiring and emotional TT",
    key="review_input",
    height=150
)

# predict button
if st.button("Sentiment Analysis"):
    if user_input.strip():
        sentiment = predict(user_input, tokenizer, model) 
        
        st.markdown("---")
        st.subheader("Sentiment Prediction:")
        
        if "FRESH" in sentiment:
            st.success(f"Review Sentiment: **{sentiment}**")
        else:
            st.error(f"Review Sentiment: **{sentiment}**")
        
        st.markdown(
            """
            *Label notes:*
            - **FRESH (POSITIVE)**: Review tends to be positive.
            - **ROTTEN (NEGATIF)**: Review tends to be negative.
            """
        )
    else:
        st.warning("Please input your review before analyzing!")

st.markdown("---")
st.caption(f"Model Inference runs on device: **{DEVICE}**")