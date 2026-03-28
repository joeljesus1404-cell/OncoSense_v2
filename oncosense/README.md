# 🧬 OncoSense — Quantum ML Platform for Early Disease Detection

**Team MindMatrix | HackHustle 2.0 | Healthcare Domain**
**SDG 3: Good Health & Well-being | SDG 10: Reduced Inequalities | SDG 9: Innovation**

---

## Abstract

OncoSense is a hybrid quantum-classical machine learning platform for early cancer detection featuring **two diagnosis modes**:

1. **Tabular Diagnosis** — Input 30 biopsy measurements, classified by Quantum Kernel SVM
2. **Image Diagnosis (Novel)** — Upload histopathology slides, processed through a **Hybrid CNN → Quantum pipeline** (ResNet18 feature extraction → PCA → ZZFeatureMap → Fidelity Quantum Kernel → SVM)

The platform is delivered as a **Streamlit web application** with PDF report generation, cross-model comparison dashboards, and ROC/confusion matrix visualizations.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Image Diagnosis Setup (Optional)

```bash
# 1. Download BreakHis dataset from Kaggle:
#    https://www.kaggle.com/datasets/ambarish/breakhis

# 2. Organize images:
#    data/breakhis/benign/     (benign images)
#    data/breakhis/malignant/  (malignant images)

# 3. Train the hybrid pipeline:
python train_image_model.py --data_dir ./data/breakhis --output_dir ./models
```

---

## Architecture

### Hybrid CNN → Quantum Pipeline (Novel Contribution)
```
Histopathology Image
        ↓
  ResNet18 (frozen, pretrained)
        ↓
  512-dim Feature Vector
        ↓
  MinMaxScaler + PCA (512 → 4)
        ↓
  ZZFeatureMap (4 qubits, 2 reps, full entanglement)
        ↓
  Fidelity Quantum Kernel: K(x,x') = |⟨ψ(x)|ψ(x')⟩|²
        ↓
  SVC (precomputed kernel)
        ↓
  Malignant / Benign + Confidence Score
```

### Tabular Pipeline
```
30 Biopsy Features → MinMaxScaler → PCA (30 → 4) → ZZFeatureMap → Quantum Kernel → SVM
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| CNN Feature Extraction | PyTorch / ResNet18 | Deep feature extraction from images |
| Quantum Computing | Qiskit / Qiskit ML | ZZFeatureMap, Fidelity Quantum Kernel |
| Quantum Simulator | Qiskit Aer | Statevector simulator |
| Classical ML | Scikit-learn | SVM baselines, PCA, evaluation |
| Data Processing | Pandas, NumPy | ETL pipeline |
| Visualization | Plotly, Matplotlib, Seaborn | ROC, confusion matrices, charts |
| Web Dashboard | Streamlit | Doctor-facing UI |
| Reports | FPDF2 | Patient diagnosis PDF generation |

---

## Project Structure

```
oncosense/
├── app.py                              # Main Streamlit app (5 pages)
├── train_image_model.py                # Hybrid pipeline training script
├── requirements.txt                    # Python dependencies
├── railway.json                        # Railway deployment config
├── .python-version                     # Python version for deployment
├── README.md
└── utils/
    ├── preprocessing.py                # Tabular data pipeline
    ├── quantum_engine.py               # Quantum Kernel SVM (Qiskit)
    ├── classical_engine.py             # Classical SVM baselines
    ├── image_feature_extractor.py      # ResNet18 CNN feature extractor
    ├── hybrid_quantum_pipeline.py      # Hybrid CNN→Quantum pipeline
    └── report_generator.py             # PDF diagnosis reports
```

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Breast Cancer — Tabular + Image Hybrid Quantum | ✅ Current |
| Phase 2 | Multi-Disease (Diabetes, Heart, Lung Cancer) | 🔮 Planned |
| Phase 3 | Federated Quantum Learning + EHR Integration | 🚀 Future |

---

*OncoSense — Where Quantum Meets Care*

HackHustle 2.0 | Saveetha Engineering College, Chennai | April 16-17, 2026
