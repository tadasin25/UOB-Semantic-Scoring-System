import streamlit as st
import pandas as pd
import torch
from io import BytesIO
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Set UOB-style page config
st.set_page_config(
    page_title="UOB Semantic Scoring System",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# UOB Styling
UOB_BLUE = "#002D72"
st.markdown(f"""
    <style>
    .reportview-container {{
        background-color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }}
    h1, h2, h3, label, .stButton>button {{
        color: {UOB_BLUE};
    }}
    .stButton>button {{
        background-color: {UOB_BLUE};
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        font-size: 16px;
    }}
    .stButton>button:hover {{
        background-color: #004C97;
        color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💬 UOB Semantic Scoring System</h1>", unsafe_allow_html=True)
st.write("Upload an Excel file with `chatbot_answer` and `reference_answer` columns to get similarity scoring.")

# Load models once
@st.cache_resource
def load_models():
    dense = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    nli_pipe = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    return dense, nli_pipe

dense_model, nli = load_models()

# Sidebar thresholds
st.sidebar.header("⚙️ Threshold Settings")
fully_correct_threshold = st.sidebar.slider(
    "Fully Correct Threshold (Cosine ≥)", min_value=0.0, max_value=1.0, value=0.85, step=0.01
)
partly_correct_threshold = st.sidebar.slider(
    "Partly Correct Threshold (Cosine ≥)", min_value=0.0, max_value=1.0, value=0.65, step=0.01
)

# Scoring logic (with dynamic thresholds)
def evaluate_answers(chatbot, reference, fully_thresh, partly_thresh):
    try:
        emb_chatbot = dense_model.encode(chatbot, convert_to_tensor=True)
        emb_ref = dense_model.encode(reference, convert_to_tensor=True)
        sim_score = torch.nn.functional.cosine_similarity(emb_chatbot, emb_ref, dim=0).item()

        nli_result = nli(f"{reference} </s> {chatbot}")
        label = nli_result[0]['label'].lower()

        if label == "contradiction" or sim_score < partly_thresh:
            judgment = "❌ Incorrect"
        elif label == "entailment" and sim_score >= fully_thresh:
            judgment = "✅ Fully correct"
        else:
            judgment = "🟡 Partly correct"

        return round(sim_score, 4), label, judgment
    except Exception as e:
        return None, "error", str(e)

# Upload file
uploaded_file = st.file_uploader("📤 Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if 'chatbot_answer' not in df.columns or 'reference_answer' not in df.columns:
            st.error("❌ Missing required columns: 'chatbot_answer' and 'reference_answer'")
        else:
            with st.spinner("🔍 Evaluating answers..."):
                results = df.apply(
                    lambda row: evaluate_answers(
                        row['chatbot_answer'],
                        row['reference_answer'],
                        fully_correct_threshold,
                        partly_correct_threshold
                    ),
                    axis=1
                )
                df[['cosine_score', 'nli_label', 'judgment']] = pd.DataFrame(results.tolist(), index=df.index)

            st.success("✅ Scoring complete!")
            st.dataframe(df)

            # Download button
            output = BytesIO()
            df.to_excel(output, index=False)
            st.download_button(
                label="📥 Download Scored Excel",
                data=output.getvalue(),
                file_name="scored_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
