"""
OncoSense - Quantum ML Platform for Early Disease Detection
Main Streamlit Application

Team MindMatrix | HackHustle 2.0 | Healthcare Domain
SDG 3: Good Health & Well-being | SDG 10: Reduced Inequalities
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import load_data, get_dataset_info, preprocess_data, preprocess_patient_input
from utils.quantum_engine import train_quantum_svm, predict_single_quantum, get_circuit_info
from utils.classical_engine import train_all_classical, predict_single_classical
from utils.report_generator import generate_report

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="OncoSense | Quantum Cancer Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #4B0082 0%, #7B2FBE 50%, #9B59B6 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(75, 0, 130, 0.3);
    }

    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(75, 0, 130, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #9B59B6;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .diagnosis-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }

    .malignant {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
    }

    .benign {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
    }

    .diagnosis-label {
        font-size: 2.5rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .confidence-text {
        font-size: 1.3rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }

    .processing-step {
        display: flex;
        align-items: center;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        background: rgba(75, 0, 130, 0.1);
        border-left: 3px solid #9B59B6;
    }

    .sdg-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .sdg-3 { background: #4C9F38; color: white; }
    .sdg-9 { background: #FD6925; color: white; }
    .sdg-10 { background: #DD1367; color: white; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    .quantum-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE ====================
def init_session_state():
    defaults = {
        'models_trained': False,
        'quantum_results': None,
        'classical_results': None,
        'preprocessed': None,
        'dataset_info': None,
        'diagnosis_result': None,
        'patient_features': None,
        'current_page': 'home',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ==================== HELPER FUNCTIONS ====================
def display_metric_card(label, value, prefix="", suffix=""):
    """Display a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def train_models(quantum_train_size=100, quantum_test_size=30):
    """
    Train all models (quantum + classical) and cache results.
    
    Quantum kernel computation scales O(n²) — computing the full 455×455 kernel matrix
    takes significant time on a simulator. We use a stratified subset for quantum training
    while classical models train on the full dataset.
    
    Parameters:
        quantum_train_size: Number of training samples for quantum kernel (default 100)
        quantum_test_size: Number of test samples for quantum evaluation (default 30)
    """
    # Load and preprocess
    df, raw_data = load_data()
    dataset_info = get_dataset_info(df)
    preprocessed = preprocess_data(df, n_components=4)

    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']

    # Classical SVMs — train on FULL dataset
    classical_results = train_all_classical(X_train, y_train, X_test, y_test)

    # Quantum SVM — train on subset for practical computation time
    # Stratified sampling to maintain class balance
    from sklearn.model_selection import train_test_split as tts
    q_train_size = min(quantum_train_size, len(X_train))
    q_test_size = min(quantum_test_size, len(X_test))

    if q_train_size < len(X_train):
        X_q_train, _, y_q_train, _ = tts(
            X_train, y_train, train_size=q_train_size, 
            random_state=42, stratify=y_train
        )
    else:
        X_q_train, y_q_train = X_train, y_train

    if q_test_size < len(X_test):
        X_q_test, _, y_q_test, _ = tts(
            X_test, y_test, train_size=q_test_size,
            random_state=42, stratify=y_test
        )
    else:
        X_q_test, y_q_test = X_test, y_test

    quantum_results = train_quantum_svm(
        X_q_train, y_q_train, X_q_test, y_q_test, n_features=4, reps=2
    )

    # Store the quantum training data for later single predictions
    preprocessed['X_q_train'] = X_q_train

    return dataset_info, preprocessed, quantum_results, classical_results


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #9B59B6; font-weight: 800; font-size: 1.8rem;">🧬 OncoSense</h2>
        <p style="color: #888; font-size: 0.85rem;">Quantum ML Platform for<br>Early Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Tabular Diagnosis", "🖼️ Image Diagnosis", "📊 Model Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    # Quantum Config
    st.markdown("**⚛️ Quantum Config**")
    quantum_train_size = st.slider(
        "Quantum training samples",
        min_value=50, max_value=455, value=100, step=25,
        help="Quantum kernel scales O(n²). Use 80-120 for demo speed, full 455 for max accuracy."
    )

    st.divider()

    # SDG Badges
    st.markdown("""
    <div style="text-align: center;">
        <span class="sdg-badge sdg-3">SDG 3: Good Health</span><br>
        <span class="sdg-badge sdg-10">SDG 10: Reduced Inequalities</span><br>
        <span class="sdg-badge sdg-9">SDG 9: Innovation</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Team MindMatrix | HackHustle 2.0")
    st.caption("SRM Valliammai Engineering College")


# ==================== HOME PAGE ====================
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 OncoSense</h1>
        <p>Quantum Machine Learning Platform for Early Disease Detection</p>
        <p style="font-size: 0.9rem; opacity: 0.7; margin-top: 1rem;">
            Team MindMatrix • HackHustle 2.0 • Saveetha Engineering College, Chennai
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_card("Dataset Size", "569", suffix=" samples")
    with col2:
        display_metric_card("Diagnosis Modes", "2", suffix=" (Tabular + Image)")
    with col3:
        display_metric_card("Quantum Qubits", "4", suffix=" qubits")
    with col4:
        display_metric_card("CNN Backbone", "ResNet", suffix="18")

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("### ⚙️ How OncoSense Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔬 Mode 1: Tabular Diagnosis**

        Doctor enters 30 standard tumor measurements from a biopsy report.
        Data flows through: MinMaxScaler → PCA (30→4) → ZZFeatureMap →
        Fidelity Quantum Kernel → SVM Classification.

        **🖼️ Mode 2: Image Diagnosis (Hybrid CNN → Quantum)**

        Doctor uploads a histopathology slide image. A pretrained **ResNet18**
        extracts 512 deep features → PCA reduces to 4 → same Quantum Kernel
        pipeline classifies. This is a **novel hybrid architecture**.
        """)

    with col2:
        st.markdown("""
        **🏥 Clinical Workflow**

        Designed for doctors in rural and resource-limited hospitals:

        1. **Input**: Biopsy numbers OR histopathology slide image
        2. **Process**: Quantum pipeline preprocesses, encodes, and classifies
        3. **Output**: Instant diagnosis (Malignant/Benign) with confidence score
        4. **Report**: Download professional PDF diagnosis report

        **Cost**: Near-zero per prediction vs INR 5,000-50,000 for traditional cycles.
        **Speed**: Seconds vs 3-7 days for lab-based diagnosis.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Development Roadmap
    st.markdown("### 🗺️ Development Roadmap")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #38ef7d; font-weight: 700; font-size: 1.2rem;">📌 Phase 1 — Current</div>
            <div style="color: #ccc; margin-top: 0.5rem; font-size: 0.9rem;">
                Breast Cancer Detection<br>
                Tabular + Image Diagnosis<br>
                Hybrid CNN-Quantum Pipeline<br>
                Streamlit Dashboard + PDF Reports
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #667eea; font-weight: 700; font-size: 1.2rem;">🔮 Phase 2 — Expansion</div>
            <div style="color: #ccc; margin-top: 0.5rem; font-size: 0.9rem;">
                Diabetes (Pima Indians)<br>
                Heart Disease (Cleveland UCI)<br>
                Lung Cancer (UCI Lung)<br>
                Multi-Disease Quantum Engine
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #ff416c; font-weight: 700; font-size: 1.2rem;">🚀 Phase 3 — Federated</div>
            <div style="color: #ccc; margin-top: 0.5rem; font-size: 0.9rem;">
                Federated Quantum Learning<br>
                Cross-Hospital Training<br>
                EHR Integration<br>
                Mobile Point-of-Care App
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==================== DIAGNOSIS PAGE (TABULAR) ====================
elif page == "🔬 Tabular Diagnosis":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Patient Diagnosis</h1>
        <p>Enter 30 tumor biopsy measurements for quantum-enhanced cancer classification</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Train models if not already done
    if not st.session_state.models_trained:
        st.markdown("### ⚡ Initialize Quantum Pipeline")
        st.info("Models need to be trained first. This initializes the Quantum Kernel SVM and Classical baselines.")

        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            with st.spinner(""):
                progress = st.progress(0)
                status = st.empty()

                status.markdown('<div class="processing-step">✅ Loading UCI Wisconsin Breast Cancer Dataset...</div>', unsafe_allow_html=True)
                progress.progress(10)
                time.sleep(0.3)

                status.markdown('<div class="processing-step">✅ Preprocessing: MinMaxScaler normalization...</div>', unsafe_allow_html=True)
                progress.progress(20)

                status.markdown('<div class="processing-step">✅ PCA Dimensionality Reduction (30 → 4 features)...</div>', unsafe_allow_html=True)
                progress.progress(30)

                status.markdown('<div class="processing-step">⚛️ Building ZZFeatureMap quantum circuit (4 qubits, 2 reps)...</div>', unsafe_allow_html=True)
                progress.progress(40)

                status.markdown('<div class="processing-step">⚛️ Computing Fidelity Quantum Kernel matrix...</div>', unsafe_allow_html=True)
                progress.progress(50)

                # Actual training
                try:
                    dataset_info, preprocessed, quantum_results, classical_results = train_models(
                        quantum_train_size=quantum_train_size
                    )

                    status.markdown('<div class="processing-step">✅ Quantum Kernel SVM trained successfully!</div>', unsafe_allow_html=True)
                    progress.progress(80)

                    status.markdown('<div class="processing-step">✅ Classical SVMs (Linear + RBF) trained!</div>', unsafe_allow_html=True)
                    progress.progress(100)

                    # Store in session
                    st.session_state.models_trained = True
                    st.session_state.quantum_results = quantum_results
                    st.session_state.classical_results = classical_results
                    st.session_state.preprocessed = preprocessed
                    st.session_state.dataset_info = dataset_info

                    st.success("All models trained! Scroll down to enter patient data.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    st.exception(e)

    # Step 2: Patient input form
    if st.session_state.models_trained:

        # Show model status
        qm = st.session_state.quantum_results['metrics']
        st.success(f"✅ Models ready — Quantum SVM Accuracy: **{qm['accuracy']:.2%}** | AUC: **{qm['auc_roc']:.4f}**")

        st.markdown("### 📋 Patient Biopsy Data Input")
        st.caption("Enter the 30 standard tumor measurements from the biopsy report. All values are numeric.")

        # Load feature names from sklearn
        from sklearn.datasets import load_breast_cancer
        feature_names = load_breast_cancer().feature_names

        # Group features into Mean, SE, Worst
        mean_features = [f for f in feature_names if 'mean' in f]
        se_features = [f for f in feature_names if 'error' in f]
        worst_features = [f for f in feature_names if 'worst' in f]

        # Example data (first malignant sample from dataset)
        example_data = load_breast_cancer().data[0]
        example_dict = dict(zip(feature_names, example_data))

        use_example = st.checkbox("📌 Load example data (first dataset sample) for testing", value=False)

        patient_features = {}

        tab1, tab2, tab3 = st.tabs(["📐 Mean Features", "📏 Standard Error", "⚠️ Worst Features"])

        with tab1:
            cols = st.columns(5)
            for i, feat in enumerate(mean_features):
                with cols[i % 5]:
                    default_val = float(example_dict[feat]) if use_example else 0.0
                    patient_features[feat] = st.number_input(
                        feat.replace('mean ', '').title(),
                        value=default_val,
                        format="%.4f",
                        key=f"mean_{i}"
                    )

        with tab2:
            cols = st.columns(5)
            for i, feat in enumerate(se_features):
                with cols[i % 5]:
                    default_val = float(example_dict[feat]) if use_example else 0.0
                    patient_features[feat] = st.number_input(
                        feat.replace('mean ', '').replace(' error', ' SE').title(),
                        value=default_val,
                        format="%.4f",
                        key=f"se_{i}"
                    )

        with tab3:
            cols = st.columns(5)
            for i, feat in enumerate(worst_features):
                with cols[i % 5]:
                    default_val = float(example_dict[feat]) if use_example else 0.0
                    patient_features[feat] = st.number_input(
                        feat.replace('worst ', '').title(),
                        value=default_val,
                        format="%.4f",
                        key=f"worst_{i}"
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Run Diagnosis
        if st.button("⚛️ Run Quantum Analysis", type="primary", use_container_width=True):

            with st.spinner("Running quantum analysis..."):
                # Preprocess patient input
                X_patient = preprocess_patient_input(
                    patient_features,
                    st.session_state.preprocessed['scaler'],
                    st.session_state.preprocessed['pca']
                )

                # Quantum prediction
                q_result = predict_single_quantum(
                    X_patient,
                    st.session_state.preprocessed['X_q_train'],
                    st.session_state.quantum_results['quantum_kernel'],
                    st.session_state.quantum_results['model']
                )

                # Classical predictions
                c_linear = predict_single_classical(
                    X_patient, st.session_state.classical_results['linear']['model']
                )
                c_rbf = predict_single_classical(
                    X_patient, st.session_state.classical_results['rbf']['model']
                )

                st.session_state.diagnosis_result = q_result
                st.session_state.patient_features = patient_features

            # ==================== RESULTS DISPLAY ====================
            st.markdown("---")
            st.markdown("### 🧬 Diagnosis Result")

            # Main diagnosis box
            css_class = "malignant" if q_result['label'] == "Malignant" else "benign"
            emoji = "🔴" if q_result['label'] == "Malignant" else "🟢"

            st.markdown(f"""
            <div class="diagnosis-box {css_class}">
                <div class="diagnosis-label">{emoji} {q_result['label']}</div>
                <div class="confidence-text">Confidence: {q_result['confidence']:.2%}</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
                    Quantum Kernel SVM (ZZFeatureMap + Fidelity Kernel)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparison across models
            st.markdown("### 📊 Cross-Model Comparison")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #9B59B6; font-weight: 700;">⚛️ Quantum SVM</div>
                    <div class="metric-value">{q_result['label']}</div>
                    <div class="metric-label">Confidence: {q_result['confidence']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #667eea; font-weight: 700;">📐 SVM (Linear)</div>
                    <div class="metric-value">{c_linear['label']}</div>
                    <div class="metric-label">Confidence: {c_linear['confidence']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #38ef7d; font-weight: 700;">🌀 SVM (RBF)</div>
                    <div class="metric-value">{c_rbf['label']}</div>
                    <div class="metric-label">Confidence: {c_rbf['confidence']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge chart
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=q_result['prob_malignant'] * 100,
                title={'text': "Malignancy Probability (%)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#9B59B6"},
                    'steps': [
                        {'range': [0, 30], 'color': "#38ef7d"},
                        {'range': [30, 70], 'color': "#FFD93D"},
                        {'range': [70, 100], 'color': "#ff416c"},
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                number={'suffix': '%', 'font': {'size': 36}}
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ccc'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # PDF Report generation
            st.markdown("### 📄 Download Diagnosis Report")

            metrics_comparison = {
                "Quantum Kernel SVM": st.session_state.quantum_results['metrics'],
                "Classical SVM (Linear)": st.session_state.classical_results['linear']['metrics'],
                "Classical SVM (RBF)": st.session_state.classical_results['rbf']['metrics'],
            }

            report_path = os.path.join(os.path.dirname(__file__), "diagnosis_report.pdf")
            generate_report(
                q_result,
                metrics_comparison,
                patient_features=patient_features,
                output_path=report_path
            )

            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Download Patient Diagnosis Report (PDF)",
                    data=f,
                    file_name="OncoSense_Diagnosis_Report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )


# ==================== IMAGE DIAGNOSIS PAGE ====================
elif page == "🖼️ Image Diagnosis":
    st.markdown("""
    <div class="main-header">
        <h1>🖼️ Image-Based Diagnosis</h1>
        <p>Upload histopathology slides — CNN extracts features, Quantum Kernel classifies</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if hybrid model is available
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    hybrid_available = os.path.exists(os.path.join(model_dir, 'hybrid_pipeline.pkl'))

    if not hybrid_available:
        st.warning("⚠️ Hybrid model not trained yet. Train it first using the instructions below.")

        st.markdown("""
        ### 🛠️ How to Set Up Image-Based Diagnosis

        **Step 1 — Download the BreakHis Dataset:**

        Go to [Kaggle — BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis) and download the dataset.

        **Step 2 — Organize the images:**

        ```
        data/breakhis/
            benign/
                image1.png, image2.png, ...
            malignant/
                image1.png, image2.png, ...
        ```

        **Step 3 — Run the training script:**

        ```bash
        python train_image_model.py --data_dir ./data/breakhis --output_dir ./models
        ```

        **Step 4 — Restart the app.** The Image Diagnosis tab will auto-detect the trained model.

        ---

        **Want to try without training?** You can still upload images below — the app will use the 
        CNN to extract features and show what the pipeline does, just without quantum classification.
        """)
        st.divider()

    # ===== IMAGE UPLOAD =====
    st.markdown("### 📤 Upload Histopathology Image")
    st.caption("Upload a breast tissue biopsy slide image (.png, .jpg, .jpeg, .tiff)")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a histopathology slide image for analysis"
    )

    if uploaded_file is not None:
        from PIL import Image as PILImage
        image = PILImage.open(uploaded_file).convert('RGB')

        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Histopathology Slide", use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #9B59B6; font-weight: 700; font-size: 1.1rem;">📋 Image Details</div>
                <br>
                <div style="color: #ccc; font-size: 0.9rem;">
                    <strong>Filename:</strong> {uploaded_file.name}<br>
                    <strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB<br>
                    <strong>Dimensions:</strong> {image.size[0]} x {image.size[1]} px<br>
                    <strong>Mode:</strong> {image.mode}<br>
                    <br>
                    <strong>Pipeline:</strong><br>
                    Image → ResNet18 → 512 features → PCA → 4 features → Quantum Kernel → SVM
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Run Analysis
        if st.button("⚛️ Run Hybrid Quantum Analysis", type="primary", use_container_width=True):

            with st.spinner(""):
                progress = st.progress(0)
                status = st.empty()

                # Step 1: CNN Feature Extraction
                status.markdown('<div class="processing-step">🧠 Loading ResNet18 feature extractor...</div>', unsafe_allow_html=True)
                progress.progress(15)

                from utils.image_feature_extractor import FeatureExtractor
                extractor = FeatureExtractor(model_name='resnet18')

                status.markdown('<div class="processing-step">🔬 Extracting 512-dim deep features from image...</div>', unsafe_allow_html=True)
                progress.progress(35)

                cnn_features = extractor.extract_single(image)

                status.markdown(f'<div class="processing-step">✅ Feature extraction complete — {len(cnn_features)} features extracted</div>', unsafe_allow_html=True)
                progress.progress(50)

                if hybrid_available:
                    # Full hybrid pipeline prediction
                    status.markdown('<div class="processing-step">⚛️ Loading trained Hybrid Quantum Pipeline...</div>', unsafe_allow_html=True)
                    progress.progress(60)

                    from utils.hybrid_quantum_pipeline import HybridQuantumPipeline
                    pipeline = HybridQuantumPipeline()
                    pipeline.load(model_dir)

                    status.markdown('<div class="processing-step">⚛️ Computing quantum kernel & classifying...</div>', unsafe_allow_html=True)
                    progress.progress(80)

                    result = pipeline.predict_image(cnn_features)

                    progress.progress(100)
                    status.empty()

                    # ===== DISPLAY RESULTS =====
                    st.markdown("---")
                    st.markdown("### 🧬 Diagnosis Result (Hybrid CNN → Quantum)")

                    q = result['quantum']
                    css_class = "malignant" if q['label'] == "Malignant" else "benign"
                    emoji = "🔴" if q['label'] == "Malignant" else "🟢"

                    st.markdown(f"""
                    <div class="diagnosis-box {css_class}">
                        <div class="diagnosis-label">{emoji} {q['label']}</div>
                        <div class="confidence-text">Confidence: {q['confidence']:.2%}</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
                            ResNet18 → PCA → ZZFeatureMap → Fidelity Quantum Kernel → SVM
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Cross-model comparison
                    st.markdown("### 📊 Cross-Model Comparison")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color: #9B59B6; font-weight: 700;">⚛️ Quantum SVM</div>
                            <div class="metric-value">{q['label']}</div>
                            <div class="metric-label">Confidence: {q['confidence']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        l = result['linear']
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color: #667eea; font-weight: 700;">📐 SVM (Linear)</div>
                            <div class="metric-value">{l['label']}</div>
                            <div class="metric-label">Confidence: {l['confidence']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        r = result['rbf']
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color: #38ef7d; font-weight: 700;">🌀 SVM (RBF)</div>
                            <div class="metric-value">{r['label']}</div>
                            <div class="metric-label">Confidence: {r['confidence']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Probability gauge
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=q['prob_malignant'] * 100,
                        title={'text': "Malignancy Probability (%)", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#9B59B6"},
                            'steps': [
                                {'range': [0, 30], 'color': "#38ef7d"},
                                {'range': [30, 70], 'color': "#FFD93D"},
                                {'range': [70, 100], 'color': "#ff416c"},
                            ],
                        },
                        number={'suffix': '%', 'font': {'size': 36}}
                    ))
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#ccc'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # PDF Report
                    st.markdown("### 📄 Download Report")
                    from utils.report_generator import generate_report
                    metrics_cmp = {
                        'Quantum Kernel SVM (Hybrid)': pipeline.quantum_metrics,
                        'Classical SVM (Linear)': pipeline.linear_metrics,
                        'Classical SVM (RBF)': pipeline.rbf_metrics,
                    }
                    report_path = os.path.join(os.path.dirname(__file__), "image_diagnosis_report.pdf")
                    generate_report(q, metrics_cmp, output_path=report_path)

                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📥 Download Image Diagnosis Report (PDF)",
                            data=f,
                            file_name="OncoSense_Image_Diagnosis.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )

                else:
                    # No trained model — show feature extraction demo
                    progress.progress(100)
                    status.empty()

                    st.markdown("---")
                    st.markdown("### 🔬 CNN Feature Extraction Preview")
                    st.info("Hybrid model not trained — showing extracted features only. Train the model to get full quantum classification.")

                    # Feature visualization
                    import plotly.express as px

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_hist = px.histogram(
                            x=cnn_features, nbins=50,
                            title="ResNet18 Feature Distribution (512-dim)",
                            labels={'x': 'Feature Value', 'y': 'Count'},
                            color_discrete_sequence=['#9B59B6']
                        )
                        fig_hist.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#ccc'),
                            height=350,
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with col2:
                        # Top activated features
                        top_k = 20
                        top_indices = np.argsort(cnn_features)[-top_k:][::-1]
                        fig_top = px.bar(
                            x=[f'F{i}' for i in top_indices],
                            y=cnn_features[top_indices],
                            title=f"Top {top_k} Most Activated Features",
                            labels={'x': 'Feature Index', 'y': 'Activation'},
                            color_discrete_sequence=['#667eea']
                        )
                        fig_top.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#ccc'),
                            height=350,
                        )
                        st.plotly_chart(fig_top, use_container_width=True)

                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #9B59B6; font-weight: 700;">📊 Extraction Summary</div>
                        <div style="color: #ccc; margin-top: 0.5rem;">
                            Features extracted: <strong>{len(cnn_features)}</strong><br>
                            Non-zero features: <strong>{(cnn_features > 0).sum()}</strong><br>
                            Mean activation: <strong>{cnn_features.mean():.4f}</strong><br>
                            Max activation: <strong>{cnn_features.max():.4f}</strong><br>
                            <br>
                            <em>Train the hybrid model to get quantum classification on these features.</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        # No image uploaded — show architecture explanation
        st.markdown("### 🏗️ Hybrid CNN → Quantum Architecture")

        st.markdown("""
        This mode uses a **two-stage hybrid pipeline** — a novel approach that combines 
        the pattern recognition power of deep CNNs with the classification advantage of 
        Quantum Kernel SVMs:

        **Stage 1 — CNN Feature Extraction (Classical)**
        - A pretrained **ResNet18** processes the histopathology image
        - The final classification layer is removed
        - Output: **512-dimensional deep feature vector** capturing cell morphology, 
          tissue structure, and staining patterns

        **Stage 2 — Quantum Classification**
        - Features are normalized (MinMaxScaler) and reduced (PCA → 4 components)
        - **ZZFeatureMap** encodes features into quantum states |psi(x)>
        - **Fidelity Quantum Kernel** computes K(x,x') = |<psi(x)|psi(x')>|^2
        - **SVC** classifies using the precomputed quantum kernel matrix

        **Why this is novel:** Most quantum ML papers use tabular data. This pipeline 
        demonstrates quantum advantage on **real medical images** — the CNN handles 
        high-dimensional pixel data while the quantum kernel captures non-linear feature 
        correlations that classical SVMs miss.
        """)

        # Architecture visualization
        col1, col2, col3, col4, col5 = st.columns(5)
        steps = [
            ("🖼️", "Image\nInput", "#4B0082"),
            ("🧠", "ResNet18\n512 features", "#667eea"),
            ("📉", "PCA\n4 features", "#38ef7d"),
            ("⚛️", "Quantum\nKernel", "#9B59B6"),
            ("🎯", "Diagnosis\nResult", "#ff416c"),
        ]
        for col, (icon, label, color) in zip([col1, col2, col3, col4, col5], steps):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color: {color};">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="color: {color}; font-weight: 600; font-size: 0.85rem; margin-top: 0.5rem; white-space: pre-line;">{label}</div>
                </div>
                """, unsafe_allow_html=True)


# ==================== MODEL ANALYTICS PAGE ====================
elif page == "📊 Model Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Analytics</h1>
        <p>Quantum vs Classical — Performance Comparison & Visualizations</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Go to **Start Diagnosis** and click **Train All Models** first.")
    else:
        qr = st.session_state.quantum_results
        cr = st.session_state.classical_results
        pp = st.session_state.preprocessed

        # ===== METRICS COMPARISON TABLE =====
        st.markdown("### 📈 Performance Metrics Comparison")

        metrics_data = {
            "Model": ["⚛️ Quantum Kernel SVM", "📐 Classical SVM (Linear)", "🌀 Classical SVM (RBF)"],
            "Accuracy": [qr['metrics']['accuracy'], cr['linear']['metrics']['accuracy'], cr['rbf']['metrics']['accuracy']],
            "Precision": [qr['metrics']['precision'], cr['linear']['metrics']['precision'], cr['rbf']['metrics']['precision']],
            "Recall": [qr['metrics']['recall'], cr['linear']['metrics']['recall'], cr['rbf']['metrics']['recall']],
            "F1 Score": [qr['metrics']['f1_score'], cr['linear']['metrics']['f1_score'], cr['rbf']['metrics']['f1_score']],
            "AUC-ROC": [qr['metrics']['auc_roc'], cr['linear']['metrics']['auc_roc'], cr['rbf']['metrics']['auc_roc']],
        }

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(
            df_metrics.style.format({
                'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                'Recall': '{:.4f}', 'F1 Score': '{:.4f}', 'AUC-ROC': '{:.4f}'
            }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'], color='#4B0082'),
            use_container_width=True,
            hide_index=True
        )

        # Metric cards row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            display_metric_card("Q-SVM Accuracy", f"{qr['metrics']['accuracy']:.2%}")
        with col2:
            display_metric_card("Q-SVM Precision", f"{qr['metrics']['precision']:.2%}")
        with col3:
            display_metric_card("Q-SVM Recall", f"{qr['metrics']['recall']:.2%}")
        with col4:
            display_metric_card("Q-SVM F1", f"{qr['metrics']['f1_score']:.2%}")
        with col5:
            display_metric_card("Q-SVM AUC", f"{qr['metrics']['auc_roc']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== ROC CURVES =====
        st.markdown("### 📉 ROC Curves — Quantum vs Classical")

        fig_roc = go.Figure()

        # Quantum ROC
        fig_roc.add_trace(go.Scatter(
            x=qr['metrics']['roc_curve']['fpr'],
            y=qr['metrics']['roc_curve']['tpr'],
            mode='lines',
            name=f"Quantum SVM (AUC={qr['metrics']['auc_roc']:.4f})",
            line=dict(color='#9B59B6', width=3)
        ))

        # Linear ROC
        fig_roc.add_trace(go.Scatter(
            x=cr['linear']['metrics']['roc_curve']['fpr'],
            y=cr['linear']['metrics']['roc_curve']['tpr'],
            mode='lines',
            name=f"Linear SVM (AUC={cr['linear']['metrics']['auc_roc']:.4f})",
            line=dict(color='#667eea', width=2, dash='dash')
        ))

        # RBF ROC
        fig_roc.add_trace(go.Scatter(
            x=cr['rbf']['metrics']['roc_curve']['fpr'],
            y=cr['rbf']['metrics']['roc_curve']['tpr'],
            mode='lines',
            name=f"RBF SVM (AUC={cr['rbf']['metrics']['auc_roc']:.4f})",
            line=dict(color='#38ef7d', width=2, dash='dot')
        ))

        # Diagonal
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Baseline',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccc'),
            legend=dict(x=0.5, y=0.05),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # ===== CONFUSION MATRICES =====
        st.markdown("### 🎯 Confusion Matrices")

        col1, col2, col3 = st.columns(3)

        for col, (name, results, color) in zip(
            [col1, col2, col3],
            [
                ("Quantum Kernel SVM", qr, 'Purples'),
                ("Classical SVM (Linear)", cr['linear'], 'Blues'),
                ("Classical SVM (RBF)", cr['rbf'], 'Greens'),
            ]
        ):
            with col:
                cm = np.array(results['metrics']['confusion_matrix'])
                fig_cm, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap=color,
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'],
                    ax=ax, cbar=False,
                    annot_kws={"size": 16, "weight": "bold"}
                )
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                ax.set_title(name, fontsize=11, fontweight='bold')
                fig_cm.patch.set_alpha(0)
                ax.tick_params(colors='#ccc')
                ax.xaxis.label.set_color('#ccc')
                ax.yaxis.label.set_color('#ccc')
                ax.title.set_color('#ccc')
                st.pyplot(fig_cm)
                plt.close()

        # ===== METRICS BAR CHART =====
        st.markdown("### 📊 Metrics Bar Comparison")

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        quantum_vals = [qr['metrics']['accuracy'], qr['metrics']['precision'], qr['metrics']['recall'], qr['metrics']['f1_score'], qr['metrics']['auc_roc']]
        linear_vals = [cr['linear']['metrics']['accuracy'], cr['linear']['metrics']['precision'], cr['linear']['metrics']['recall'], cr['linear']['metrics']['f1_score'], cr['linear']['metrics']['auc_roc']]
        rbf_vals = [cr['rbf']['metrics']['accuracy'], cr['rbf']['metrics']['precision'], cr['rbf']['metrics']['recall'], cr['rbf']['metrics']['f1_score'], cr['rbf']['metrics']['auc_roc']]

        fig_bar = go.Figure(data=[
            go.Bar(name='Quantum SVM', x=metrics_names, y=quantum_vals, marker_color='#9B59B6'),
            go.Bar(name='Linear SVM', x=metrics_names, y=linear_vals, marker_color='#667eea'),
            go.Bar(name='RBF SVM', x=metrics_names, y=rbf_vals, marker_color='#38ef7d'),
        ])
        fig_bar.update_layout(
            barmode='group',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccc'),
            yaxis=dict(range=[0.8, 1.0], gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ===== PCA VARIANCE =====
        st.markdown("### 🔬 PCA Explained Variance")

        ev = pp['explained_variance']
        fig_pca = go.Figure(data=[
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(ev))],
                y=ev,
                marker_color=['#9B59B6', '#667eea', '#38ef7d', '#ff416c'][:len(ev)],
                text=[f'{v:.2%}' for v in ev],
                textposition='outside',
            )
        ])
        fig_pca.update_layout(
            yaxis_title='Explained Variance Ratio',
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccc'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_pca, use_container_width=True)
        with col2:
            total_var = sum(ev)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_var:.2%}</div>
                <div class="metric-label">Total Variance Explained</div>
                <br>
                <div style="color: #888; font-size: 0.85rem;">
                    30 features → 4 PCA components<br>
                    Retaining {total_var:.1%} of original information
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ===== QUANTUM CIRCUIT INFO =====
        st.markdown("### ⚛️ Quantum Circuit Details")

        circuit_info = get_circuit_info(qr['feature_map'])
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric_card("Qubits", circuit_info['num_qubits'])
        with col2:
            display_metric_card("Circuit Depth", circuit_info['depth'])
        with col3:
            display_metric_card("Parameters", circuit_info['num_parameters'])
        with col4:
            gates = sum(circuit_info['gate_counts'].values())
            display_metric_card("Total Gates", gates)

        # Draw the circuit
        st.markdown("**ZZFeatureMap Circuit Diagram:**")
        try:
            fig_circuit = qr['feature_map'].decompose().draw(output='mpl', style='clifford')
            st.pyplot(fig_circuit)
            plt.close()
        except Exception:
            st.code(str(qr['feature_map'].draw(output='text')), language='text')


# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About OncoSense</h1>
        <p>Where Quantum Meets Care</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🧬 What is OncoSense?

    OncoSense is a **hybrid quantum-classical machine learning platform** designed for early disease detection,
    starting with breast cancer. It leverages Qiskit's **ZZFeatureMap** and **Fidelity Quantum Kernel** to encode
    patient biopsy data into high-dimensional Hilbert space, capturing complex non-linear patterns that classical
    algorithms struggle to detect.

    ---

    ### 🔬 Technical Architecture

    **Quantum Processing Pipeline:**
    - **Feature Encoding**: ZZFeatureMap (n_qubits=4, reps=2, entanglement=full)
    - **Kernel Computation**: Fidelity Quantum Kernel — K(x,x') = |⟨ψ(x)|ψ(x')⟩|²
    - **Classification**: SVC with precomputed quantum kernel matrix
    - **Simulation**: Qiskit Aer statevector_simulator

    **Classical Baselines:**
    - SVM with Linear Kernel
    - SVM with RBF (Radial Basis Function) Kernel

    **Data Pipeline:**
    - UCI Wisconsin Breast Cancer Dataset (569 samples, 30 features)
    - MinMaxScaler normalization → PCA (30 → 4 features) → 80/20 train-test split

    ---

    ### 🎯 SDG Alignment

    | SDG | Goal | OncoSense Contribution |
    |-----|------|----------------------|
    | **SDG 3** | Good Health & Well-being | Early cancer detection saves lives; accessible diagnosis reduces mortality |
    | **SDG 10** | Reduced Inequalities | Democratizes specialist-level diagnostics for rural populations |
    | **SDG 9** | Innovation & Infrastructure | Applies quantum computing to real-world healthcare |

    ---

    ### 🛠️ Tech Stack

    | Layer | Technology | Purpose |
    |-------|-----------|---------|
    | Quantum Computing | Qiskit / Qiskit ML | ZZFeatureMap, Fidelity Kernel, Quantum SVM |
    | Quantum Simulator | Qiskit Aer | Statevector simulation |
    | Classical ML | Scikit-learn | SVM baselines, evaluation metrics |
    | Data Processing | Pandas, NumPy | Loading, cleaning, normalization |
    | Visualization | Plotly, Matplotlib, Seaborn | ROC curves, confusion matrices, charts |
    | Web Dashboard | Streamlit | Doctor-facing UI, real-time diagnosis |
    | Report Generation | FPDF2 | Patient diagnosis PDF reports |

    ---

    ### 👥 Team MindMatrix

    | Role | Responsibility |
    |------|---------------|
    | Quantum ML Engineer | Qiskit model, kernel computation, evaluation |
    | Full Stack Developer | Streamlit UI, data pipeline, deployment |

    **Institution:** SRM Valliammai Engineering College, Chennai

    ---

    *OncoSense — Where Quantum Meets Care*

    HackHustle 2.0 | Saveetha Engineering College, Chennai | April 16-17, 2026
    """)
