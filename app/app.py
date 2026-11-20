"""
Intelligent News Categorization System
Phase 8: Production Interactive Streamlit Web Application

A complete NLP application for classifying news into 5 categories:
- Politics
- Business
- Technology
- Sports
- Entertainment
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_loader import get_model_manager, CATEGORIES, CATEGORY_COLORS
from src.prediction import get_news_predictor

# Set page configuration
st.set_page_config(
    page_title="Intelligent News Categorization System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state["history"] = []

# Cached manager and predictor
@st.cache_resource
def load_system():
    manager = get_model_manager(models_dir="models")
    predictor = get_news_predictor(models_dir="models")
    return manager, predictor

manager, predictor = load_system()
available_models = manager.get_available_model_names()

# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="margin: 0; color: #1E3A8A;">📰 NewsAI</h2>
        <p style="font-size: 13px; color: #64748B; margin: 0;">Five-Class NLP Categorization</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

nav_page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📰 News Classifier", "📊 Model Comparison", "🕒 Prediction History", "🔬 NLP Architecture & Theory"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Prediction Settings")

selected_model = st.sidebar.selectbox(
    "Select Classification Model:",
    available_models,
    index=0 if "TF-IDF + Linear SVM (Calibrated)" in available_models else 0
)

rejection_threshold = st.sidebar.slider(
    "Uncertainty Rejection Threshold:",
    min_value=0.30,
    max_value=0.80,
    value=0.50,
    step=0.05,
    help="If the model's top predicted probability is below this threshold, the prediction is flagged as uncertain."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Project Documentation")

report_md_path = "PROJECT_REPORT.md"
if os.path.exists(report_md_path):
    with open(report_md_path, "r", encoding="utf-8") as f:
        st.sidebar.download_button(
            "📄 Download Report (Markdown)",
            data=f.read(),
            file_name="Intelligent_News_Categorization_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

report_html_path = "evaluation/project_report.html"
if os.path.exists(report_html_path):
    with open(report_html_path, "r", encoding="utf-8") as f:
        st.sidebar.download_button(
            "🖨️ Download Report (Print-Ready HTML)",
            data=f.read(),
            file_name="Intelligent_News_Categorization_Report.html",
            mime="text/html",
            use_container_width=True
        )

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 12px; color: #64748B;">
        <b>Supported Categories:</b><br>
        🔴 Politics &nbsp; 🔵 Business<br>
        🟣 Technology &nbsp; 🟢 Sports<br>
        🟠 Entertainment<br><br>
        <i>Strict classical NLP & neural architectures. No Transformers or external LLMs.</i>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# PAGE 1: 🏠 HOME
# =============================================================================
if nav_page == "🏠 Home":
    st.title("📰 Intelligent News Categorization System")
    st.markdown("### An End-to-End NLP & Deep Learning News Intelligence Platform")
    
    st.markdown(
        """
        Welcome to the **Intelligent News Categorization System**, a machine learning and deep learning application 
        designed to classify news headlines and full articles into exactly **five standardized categories**:
        """
    )
    
    cols = st.columns(5)
    cat_icons = {
        "Politics": ("🏛️", "#e74c3c", "Elections, parliamentary legislation, diplomacy, government policy."),
        "Business": ("📈", "#3498db", "Markets, central banks, corporate earnings, trade, economic indicators."),
        "Technology": ("💻", "#9b59b6", "Software engineering, semiconductors, cybersecurity, cloud infrastructure."),
        "Sports": ("⚽", "#2ecc71", "Tournaments, athletic matches, championships, Olympic events."),
        "Entertainment": ("🎬", "#f39c12", "Cinema, music, performing arts, box office records, cultural festivals.")
    }
    
    for i, cat in enumerate(CATEGORIES):
        icon, color, desc = cat_icons[cat]
        with cols[i]:
            st.markdown(
                f"""
                <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 15px; border-radius: 6px; height: 180px;">
                    <h3 style="margin-top: 0; color: {color};">{icon} {cat}</h3>
                    <p style="font-size: 13px; color: #475569;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown("---")
    st.subheader("🛠️ Core Technologies & Methodological Rigor")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            **NLP Preprocessing Pipeline**
            - Contraction expansion
            - Context-aware stopword filtering
            - POS-guided WordNet lemmatization
            - Uniform integer sequence padding
            """
        )
    with c2:
        st.markdown(
            """
            **Machine Learning Models**
            - TF-IDF + Logistic Regression
            - TF-IDF + Linear SVM (Calibrated)
            - Word2Vec Skip-gram + Simple RNN
            - Word2Vec Skip-gram + LSTM
            """
        )
    with c3:
        st.markdown(
            """
            **Production Reliability Safeguards**
            - Zero data leakage guarantee
            - Calibrated probabilities (`CalibratedClassifierCV`)
            - Uncertainty rejection mechanism ($T = 0.50$)
            - Sub-millisecond inference latency
            """
        )
        
    st.markdown("---")
    st.subheader("📄 Formal Project Documentation & Downloadable Reports")
    st.markdown("Download the full comprehensive research and engineering report in your preferred format:")
    
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        if os.path.exists("PROJECT_REPORT.md"):
            with open("PROJECT_REPORT.md", "r", encoding="utf-8") as f:
                st.download_button(
                    "📥 Download Complete Project Report (Markdown .MD)",
                    data=f.read(),
                    file_name="Intelligent_News_Categorization_Report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    with rep_col2:
        if os.path.exists("evaluation/project_report.html"):
            with open("evaluation/project_report.html", "r", encoding="utf-8") as f:
                st.download_button(
                    "🖨️ Download Print-Ready Project Report (HTML / PDF)",
                    data=f.read(),
                    file_name="Intelligent_News_Categorization_Report.html",
                    mime="text/html",
                    use_container_width=True
                )

    st.markdown("---")
    st.info("💡 **Quick Start:** Switch to the **📰 News Classifier** page in the sidebar to test custom news articles or choose from preset benchmark examples!")

# =============================================================================
# PAGE 2: 📰 NEWS CLASSIFIER
# =============================================================================
elif nav_page == "📰 News Classifier":
    st.title("📰 Real-Time News Classification")
    st.markdown("Paste a news headline or full article below to classify it into one of the five target categories.")

    # Sample Presets
    sample_options = {
        "Select a pre-loaded sample headline...": "",
        "[Politics] Parliamentary election debate": "The prime minister dissolved parliament early this morning and called for snap national elections following coalition collapse.",
        "[Business] Federal Reserve interest rates": "Federal Reserve officials decided to maintain benchmark interest rates unchanged citing persistent inflation pressures in corporate bond markets.",
        "[Technology] Cybersecurity cloud vulnerability": "A prominent cybersecurity research group uncovered a widespread zero-day vulnerability affecting enterprise cloud server clusters.",
        "[Sports] Continental championship qualifier": "The national soccer squad secured qualification for the continental tournament with a dramatic stoppage-time goal in the second half.",
        "[Entertainment] Award-winning film festival": "The independent psychological thriller swept the film awards ceremony taking home best picture, best director, and best cinematography.",
        "[Ambiguous] Cross-domain multi-topic news": "The tech company signed an entertainment sponsorship deal for the mobile sports video game with prominent venture capital backing.",
        "[Edge Case] Short ambiguous sentence": "It happened yesterday afternoon."
    }
    
    selected_sample = st.selectbox("Quick Test Samples:", list(sample_options.keys()))
    default_text = sample_options[selected_sample] if selected_sample != "Select a pre-loaded sample headline..." else ""
    
    user_input = st.text_area(
        "Enter News Headline or Full Article:",
        value=default_text,
        height=140,
        placeholder="Paste news headline or paragraph here (e.g., 'The central bank raised interest rates by 25 basis points...')"
    )
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("🚀 Classify News", type="primary", use_container_width=True)
    with col_info:
        model_meta = manager.get_model_info(selected_model)
        if model_meta:
            st.caption(f"**Active Model:** `{selected_model}` | {model_meta['description']}")

    if predict_btn or (user_input and selected_sample != "Select a pre-loaded sample headline..."):
        if not user_input.strip():
            st.warning("⚠️ Please enter a news article or select a pre-loaded sample.")
        else:
            with st.spinner("Analyzing text and evaluating model probabilities..."):
                res = predictor.predict(
                    raw_text=user_input,
                    model_name=selected_model,
                    rejection_threshold=rejection_threshold
                )
                
            if res["status"] == "ERROR":
                st.error(f"❌ {res['message']}")
            else:
                pred_cat = res["predicted_category"]
                confidence = res["confidence"]
                is_confident = res["is_confident"]
                color = CATEGORY_COLORS.get(pred_cat, "#333333")
                
                # Add to session history
                st.session_state["history"].insert(0, {
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Headline Snippet": (user_input[:80] + "...") if len(user_input) > 80 else user_input,
                    "Predicted Category": pred_cat,
                    "Status": "CONFIDENT" if is_confident else "UNCERTAIN",
                    "Confidence": f"{confidence*100:.1f}%",
                    "Model": selected_model
                })

                st.markdown("---")
                st.subheader("🎯 Classification Results")
                
                # Main Prediction Display
                if is_confident:
                    st.markdown(
                        f"""
                        <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: {color}; font-weight: bold;">Identified Category</span>
                                    <h1 style="margin: 4px 0 0 0; color: {color}; font-size: 32px;">{pred_cat}</h1>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 13px; color: #64748B;">Calibrated Confidence</span>
                                    <h2 style="margin: 4px 0 0 0; color: #1E293B;">{confidence*100:.1f}%</h2>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(
                        f"""
                        ⚠️ **Uncertain Prediction Warning:**  
                        The model's highest probability (**{confidence*100:.1f}%**) is below the required confidence threshold (**{rejection_threshold*100:.1f}%**).  
                        **Result:** `{res['display_prediction']}`  
                        *Reason:* {res['uncertainty_reason']}
                        """
                    )
                    
                # Probability Distribution & Explainability Columns
                res_col1, res_col2 = st.columns([3, 2])
                
                with res_col1:
                    st.markdown("#### 📊 Probability Distribution Across All Classes")
                    prob_df = pd.DataFrame([
                        {"Category": cat, "Probability (%)": prob * 100, "Color": CATEGORY_COLORS[cat]}
                        for cat, prob in res["all_probabilities"].items()
                    ]).sort_values("Probability (%)", ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    bars = ax.barh(prob_df["Category"], prob_df["Probability (%)"], color=prob_df["Color"], height=0.55)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Calibrated Probability (%)", fontsize=10)
                    ax.axvline(rejection_threshold * 100, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({int(rejection_threshold*100)}%)")
                    ax.grid(axis='x', alpha=0.25)
                    ax.legend(loc='lower right', fontsize=9)
                    for bar in bars:
                        val = bar.get_width()
                        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontsize=9, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                with res_col2:
                    st.markdown("#### 🔍 Prediction Explainability")
                    st.write(f"**Inference Latency:** `{res['latency_ms']:.2f} ms`")
                    st.write(f"**Word Count:** `{res['words_count']} words`")
                    
                    if "top_words" in res["explanation"]:
                        st.markdown("**Top Influencing Terms (TF-IDF Feature Weights):**")
                        words = res["explanation"]["top_words"]
                        if words:
                            tags_html = " ".join([f"<span style='background-color:#E2E8F0; padding:4px 8px; border-radius:4px; font-size:12px; margin-right:4px;'>{w}</span>" for w in words])
                            st.markdown(tags_html, unsafe_allow_html=True)
                        else:
                            st.write("No distinct single keywords dominated; classification was driven by aggregate phrase scores.")
                    elif "in_vocab_words" in res["explanation"]:
                        st.markdown(f"**Sequence Tokens Processed:** `{res['explanation']['token_count']}`")
                        st.markdown("**Recognized Word2Vec Tokens:**")
                        w_list = res["explanation"]["in_vocab_words"]
                        st.code(", ".join(w_list) if w_list else "None")
                        if res["explanation"]["oov_words"]:
                            st.markdown("**Out-of-Vocabulary (OOV) Tokens:**")
                            st.code(", ".join(res["explanation"]["oov_words"]))

# =============================================================================
# PAGE 3: 📊 MODEL COMPARISON DASHBOARD
# =============================================================================
elif nav_page == "📊 Model Comparison":
    st.title("📊 Model Comparison & Benchmark Dashboard")
    st.markdown("Comprehensive evaluation metrics obtained during Phase 7 testing on **1,811 unseen test articles** and **50 out-of-distribution 2024 articles**.")
    
    # Summary Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Model", "Linear SVM", "Calibrated Platt Probabilities")
    c2.metric("Best Test Accuracy", "83.77%", "+42.96% vs RNN")
    c3.metric("Best Macro F1", "0.8376", "Optimal Balance")
    c4.metric("Inference Latency", "0.01 ms", "115x Faster than LSTM")
    
    st.markdown("---")
    
    # Comparison Table
    st.subheader("1. Detailed Performance Metrics (1,811 Test Set Articles)")
    benchmark_data = pd.DataFrame([
        {
            "Model Architecture": "TF-IDF + Linear SVM (Calibrated)",
            "Test Accuracy": "83.77%",
            "Macro Precision": "0.8389",
            "Macro Recall": "0.8370",
            "Macro F1-Score": "0.8376",
            "Weighted F1": "0.8380",
            "Inference Latency": "0.01 ms",
            "Model Size": "0.76 MB",
            "Rank": "🥇 Rank 1"
        },
        {
            "Model Architecture": "TF-IDF + Logistic Regression",
            "Test Accuracy": "83.49%",
            "Macro Precision": "0.8367",
            "Macro Recall": "0.8343",
            "Macro F1-Score": "0.8351",
            "Weighted F1": "0.8353",
            "Inference Latency": "0.00 ms",
            "Model Size": "0.76 MB",
            "Rank": "🥈 Rank 2"
        },
        {
            "Model Architecture": "Word2Vec + LSTM",
            "Test Accuracy": "40.81%",
            "Macro Precision": "0.7082",
            "Macro Recall": "0.4119",
            "Macro F1-Score": "0.3366",
            "Weighted F1": "0.3349",
            "Inference Latency": "0.92 ms",
            "Model Size": "21.35 MB",
            "Rank": "🥉 Rank 3"
        },
        {
            "Model Architecture": "Word2Vec + Simple RNN",
            "Test Accuracy": "37.16%",
            "Macro Precision": "0.5377",
            "Macro Recall": "0.3814",
            "Macro F1-Score": "0.2828",
            "Weighted F1": "0.2785",
            "Inference Latency": "0.49 ms",
            "Model Size": "6.82 MB",
            "Rank": "Rank 4"
        }
    ])
    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("2. Visual Performance Benchmarks")
    
    # Display saved charts
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("**Accuracy, Macro F1 & Latency Comparison:**")
        if os.path.exists("evaluation/model_comparison_metrics.png"):
            st.image("evaluation/model_comparison_metrics.png", use_container_width=True)
            
    with chart_cols[1]:
        st.markdown("**Per-Category F1-Score Breakdown:**")
        if os.path.exists("evaluation/per_category_comparison.png"):
            st.image("evaluation/per_category_comparison.png", use_container_width=True)
            
    st.markdown("---")
    st.subheader("3. Confusion Matrix Analysis (All 4 Models)")
    if os.path.exists("evaluation/all_models_confusion_matrices.png"):
        st.image("evaluation/all_models_confusion_matrices.png", use_container_width=True)
        
    st.markdown("---")
    st.subheader("4. External Unseen Generalization Benchmark (50 Articles from 2024)")
    ext_data = pd.DataFrame([
        {"Model": "TF-IDF + Logistic Regression", "External Accuracy": "92.00%", "External Macro F1": "0.9223", "Assessment": "Outstanding real-world generalization."},
        {"Model": "TF-IDF + Linear SVM (Calibrated)", "External Accuracy": "88.00%", "External Macro F1": "0.8850", "Assessment": "Strong discriminative generalization."},
        {"Model": "Word2Vec + LSTM", "External Accuracy": "40.00%", "External Macro F1": "0.2303", "Assessment": "Moderate; struggles on short headlines."},
        {"Model": "Word2Vec + Simple RNN", "External Accuracy": "36.00%", "External Macro F1": "0.2095", "Assessment": "Recency bias leads to category collapse."}
    ])
    st.dataframe(ext_data, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 4: 🕒 PREDICTION HISTORY
# =============================================================================
elif nav_page == "🕒 Prediction History":
    st.title("🕒 Session Prediction History")
    st.markdown("Audit recent classifications executed during this interactive browser session. *(Note: Content is stored strictly in ephemeral session memory and is never permanently written to disk)*")
    
    history_list = st.session_state["history"]
    
    if not history_list:
        st.info("ℹ️ No predictions recorded yet in this session. Go to the **📰 News Classifier** page and classify an article to see history here.")
    else:
        df_hist = pd.DataFrame(history_list)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        
        btn_c1, btn_c2 = st.columns([1, 5])
        with btn_c1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()
        with btn_c2:
            csv_data = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History as CSV",
                data=csv_data,
                file_name="news_classification_session_history.csv",
                mime="text/csv"
            )

# =============================================================================
# PAGE 5: 🔬 NLP ARCHITECTURE & THEORY
# =============================================================================
elif nav_page == "🔬 NLP Architecture & Theory":
    st.title("🔬 NLP Concepts & Theoretical Walkthrough")
    st.markdown("Detailed educational breakdown of the four NLP and Deep Learning models implemented across Phases 1 through 7.")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. TF-IDF & Classical ML",
        "2. Dense Word2Vec Embeddings",
        "3. Recurrent Neural Networks & Vanishing Gradients",
        "4. LSTM & Constant Error Carousel"
    ])
    
    with tab1:
        st.subheader("TF-IDF (Term Frequency - Inverse Document Frequency)")
        st.markdown(
            """
            TF-IDF reflects how important a word is to a document in a collection or corpus:
            $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
            Where:
            $$\\text{IDF}(t, D) = \\log \\left( \\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|} \\right) + 1$$
            
            **Why Classical Linear Models Succeeded:**
            - **Linear SVM:** Finds the hyperplane that maximizes the margin between categories in a 20,000-dimensional sparse feature space.
            - **Global Representation:** Unigrams and bigrams across the entire document are evaluated simultaneously without temporal forgetting or sequence cutoffs.
            """
        )
        
    with tab2:
        st.subheader("Word2Vec: Continuous Bag-of-Words vs. Skip-gram")
        st.markdown(
            """
            Unlike one-hot encodings which treat all words as orthogonal vectors ($v_i \\cdot v_j = 0$), **Word2Vec** learns dense distributed representations:
            - **Skip-gram (`sg=1`):** Predicts surrounding context words given a target center word $w_t$.
            - **100 Dimensions:** Captures nuanced semantic relationships (e.g. $\\text{vector('senator')} \\approx \\text{vector('lawmaker')}$) learned strictly on our 8,450 training articles.
            """
        )
        
    with tab3:
        st.subheader("Simple RNN & The Vanishing Gradient Problem")
        st.markdown(
            """
            In an ordinary RNN, the hidden state update is strictly multiplicative:
            $$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
            
            During Backpropagation Through Time (BPTT), computing the gradient of early words involves:
            $$\\frac{\\partial h_t}{\\partial h_1} = \\prod_{j=2}^t \\text{diag}(1 - h_j^2) W_{hh}^T$$
            Because $|\\tanh'(z)| \\le 1$, gradients decay exponentially towards zero across 150 tokens, causing **recency bias** where the model forgets early thematic keywords.
            """
        )
        
    with tab4:
        st.subheader("LSTM Gating & The Constant Error Carousel")
        st.markdown(
            """
            LSTM overcomes vanishing gradients by separating memory into two distinct paths:
            - **Cell State ($C_t$):** Long-term additive highway.
            - **Hidden State ($h_t$):** Short-term output.
            
            **The Four Gating Formulations:**
            1. **Forget Gate:** $f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$
            2. **Input Gate:** $i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$
            3. **Candidate State:** $\\tilde{C}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$
            4. **Additive Cell Update:** $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$
            5. **Output Gate & Hidden State:** $o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$, &nbsp; $h_t = o_t \\odot \\tanh(C_t)$
            
            Because $\\frac{\\partial C_t}{\\partial C_{t-1}} = f_t$, when $f_t \\approx 1$, error gradients flow backward through hundreds of steps without decaying!
            """
        )
