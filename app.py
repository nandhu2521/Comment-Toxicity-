import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Comment Toxicity Analysis Dashboard", layout="wide", page_icon="🛡️")

# --- CSS / STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f1f3f6; border-radius: 10px 10px 0 0; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #6366f1; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border: 1px solid #e5e7eb; }
    .verdict-box { padding: 20px; border-radius: 10px; margin-top: 20px; font-weight: bold; font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS (Synced with Data.ipynb) ---
MAX_LEN = 150 
TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)      
    text = re.sub(r'[^a-z\s]', '', text)     
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

@st.cache_resource
def load_all():
    loaded_model, loaded_tokenizer = None, None
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        h5_path = os.path.join(base_path, 'best_toxicity_model.h5')
        keras_path = os.path.join(base_path, 'models', 'lstm_model.keras')
        tok_root = os.path.join(base_path, 'tokenizer.pkl')
        tok_models = os.path.join(base_path, 'models', 'tokenizer.pkl')

        # Load Model
        if os.path.exists(h5_path):
            loaded_model = tf.keras.models.load_model(h5_path)
        elif os.path.exists(keras_path):
            loaded_model = tf.keras.models.load_model(keras_path)
            
        # Load Tokenizer
        if os.path.exists(tok_root):
            with open(tok_root, 'rb') as f:
                loaded_tokenizer = pickle.load(f)
        elif os.path.exists(tok_models):
            with open(tok_models, 'rb') as f:
                loaded_tokenizer = pickle.load(f)
                
        if loaded_model is None:
            return "Missing Model File (.h5)", None
        if loaded_tokenizer is None:
            return None, "Missing Tokenizer (.pkl)"
            
        return loaded_model, loaded_tokenizer
    except Exception as e:
        return f"System Error: {e}", None

def predict_logic(text_list, model, tokenizer):
    cleaned = [clean_text(t) for t in text_list]
    seqs = tokenizer.texts_to_sequences(cleaned)
    pad = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    
    probs = model.predict(pad)
    return probs

# --- INITIALIZATION ---
best_model, tokenizer = None, None
load_error = None

init_res = load_all()
if isinstance(init_res[0], str) and ("Error" in init_res[0] or "Missing" in init_res[0]):
    load_error = init_res[0]
elif init_res[1] is None and init_res[0] is None:
    load_error = "Files not found"
else:
    best_model, tokenizer = init_res

# --- SIDEBAR ---
with st.sidebar:
    st.divider()
    st.subheader("📖 Sample Test Cases")
    sample_comments = {
        "Custom": "",
        "Clean": "This is a very helpful explanation, thank you!",
        "Toxic": "You are an absolute idiot and I hate you.",
        "Obscene": "Shut the f*** up and go away.",
        "Threat": "I will find you and I will hurt you.",
        "Identity Hate": "Stop being so racist, it's disgusting."
    }
    sample_choice = st.selectbox("Pick a sample:", list(sample_comments.keys()))

# --- MAIN UI ---
st.title("🚨 Comment Toxicity Analysis Dashboard")

# Block UI if components are missing
if not hasattr(best_model, 'predict') or tokenizer is None:
    st.warning("⚠️ Application is in standby mode. Please fix the missing files shown in the sidebar.")
else:
    tab1, tab2 = st.tabs(["✍️ Real-time Analysis", "📈 Model Insight"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        input_text = st.text_area("✍️ Enter a comment to analyze:", value=sample_comments[sample_choice], height=150, placeholder="Type your comment here...")
        
        if st.button("Analyze Content"):
            if input_text:
                with st.spinner("Processing..."):
                    probs = predict_logic([input_text], best_model, tokenizer)[0]
                    
                    toxic_flags = [TARGET_COLS[i] for i, p in enumerate(probs) if p > 0.5]
                    warning_flags = [TARGET_COLS[i] for i, p in enumerate(probs) if 0.2 < p <= 0.5]
                    
                    st.divider()
                    st.subheader("🏁 Final Prediction")
                    
                    if toxic_flags:
                        st.error(f"🚩 **TOXIC CONTENT DETECTED**")
                        st.write(f"**Detected Flags:** {', '.join([f'`{f}`' for f in toxic_flags])}")
                    elif warning_flags:
                        st.warning(f"⚠️ **SUSPICIOUS CONTENT ALERT (Moderate Risk)**")
                        st.write(f"**Potential Categories:** {', '.join([f'`{f}`' for f in warning_flags])}")
                        st.info("The model is suspicious but not fully confident.")
                    else:
                        st.success("✅ **CLEAN CONTENT**")

                    st.divider()
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.subheader("📊 Prediction Details")
                        def get_verdict(p):
                            if p > 0.5: return '🚩 Toxic'
                            if p > 0.2: return '⚠️ Warning'
                            return '✅ Clean'
                        
                        res_df = pd.DataFrame({
                            'Category': TARGET_COLS,
                            'Confidence Score': [f"{p*100:.2f}%" for p in probs],
                            'Verdict': [get_verdict(p) for p in probs]
                        })
                        st.table(res_df)
                    
                    with c2:
                        st.subheader("📉 Score Distribution")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(x=TARGET_COLS, y=probs, palette='magma', ax=ax)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel("Probability Score")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

    # --- TAB 2: SELECTION LOGIC ---
    with tab2:
        st.header("📈 Best Model Selection Logic")
        st.markdown("""
        <div style="background-color: white; padding: 30px; border-radius: 15px; border-left: 8px solid #6366f1; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <h2 style="color: #1e293b; margin-top: 0;">🏆 WINNING MODEL: LSTM</h2>
            <hr style="border: 1px solid #e2e8f0; margin: 20px 0;">
            <p style="font-size: 1.1rem; color: #475569;">
                Based on extensive testing, the <b>LSTM (Long Short-Term Memory)</b> model was selected for its superior ability to capture word dependencies, yielding a final <b>Mean Accuracy of 0.9739</b> compared to 0.9712 for the CNN.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        # Metric Cards Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-card"><p style="color: #64748b; font-weight: 600;">📊 CNN Mean Accuracy</p><h2 style="color: #ff9f43;">0.9712</h2></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card"><p style="color: #64748b; font-weight: 600;">📉 LSTM Mean Accuracy</p><h2 style="color: #0fbcf9;">0.9739</h2></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card"><p style="color: #64748b; font-weight: 600;">🏆 Winner</p><h2 style="color: #10b981;">LSTM</h2></div>', unsafe_allow_html=True)

        st.divider()

        st.divider()
        st.subheader("📊 Category-wise Performance Comparison")
        
        # Performance data from notebook results
        comparison_data = pd.DataFrame({
            'Category': TARGET_COLS,
            'CNN Accuracy': [0.9724, 0.9883, 0.9866, 0.9430, 0.9802, 0.9567],
            'LSTM Accuracy': [0.9767, 0.9883, 0.9858, 0.9514, 0.9808, 0.9607]
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(TARGET_COLS))
        width = 0.35
        
        ax.bar(x - width/2, comparison_data['CNN Accuracy'], width, label='CNN', color='#ff9f43')
        ax.bar(x + width/2, comparison_data['LSTM Accuracy'], width, label='LSTM', color='#0fbcf9')
        
        ax.set_ylabel('Accuracy Score')
        ax.set_title('CNN vs LSTM Performance by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_COLS, rotation=45)
        ax.set_ylim(0.9, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

        st.divider()
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            with st.expander("🏗️ Architecture Deep Dive", expanded=True):
                st.markdown("""
                **LSTM Model (The Winner):**
                - **Input:** 150 Sequence length
                - **Embedding:** 128-dimensional word vectors
                - **Dropout:** SpatialDropout1D (0.3) for regulation
                - **Core:** **Bidirectional LSTM** (64 units)
                - **Head:** GlobalMaxPooling → Dense (128) → Dense (6)
                
                **Why LSTM won?**
                The Bidirectional LSTM processes text in both directions, allowing it to understand the **context** of words based on what came before *and* after them.
                """)
                
        with col_tech2:
            with st.expander("🛠️ Preprocessing Pipeline"):
                st.markdown(f"""
                **Step-by-Step Processing:**
                1. **Regex Cleaning:** Removing URLs and special characters.
                2. **Tokenization:** Text converted to integers (Vocab size: **{20000}**).
                3. **Padding:** All sequences standardized to **{MAX_LEN}** tokens.
                4. **OOV Handling:** Unknown words are mapped to a special `<OOV>` token.
                
                *This ensures the model focuses on semantic meaning rather than noise.*
                """)

        st.divider()
        st.subheader("🎯 Training Strategy")
        st_c1, st_c2, st_c3 = st.columns(3)
        with st_c1:
            st.info("**Optimizer: Adam**\n\nEfficiently manages learning rate updates.")
        with st_c2:
            st.info("**Loss: Binary Crossentropy**\n\nIdeal for multi-label classification tasks.")
        with st_c3:
            st.info("**Callback: EarlyStopping**\n\nPrevents overfitting by monitoring validation performance.")

        st.info("💡 **Project Goal:** This dashboard translates complex neural network weights into understandable safety scores, helping moderators make faster, data-driven decisions.")
