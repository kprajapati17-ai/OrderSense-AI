# OrderSense-AI: E-Commerce Order Intelligence & Support System

An end-to-end multi-modal machine learning, computer vision, and AI agent platform designed for e-commerce order intelligence. **OrderSense-AI** predicts return risk using an ensemble ML pipeline, classifies product images using a trained **TensorFlow Lite (TFLite)** model, and delivers contextual customer support resolutions via Streamlit and the Mistral AI engine.

---

## 📌 Architecture Overview

```text
Customer Order Telemetry + Product Image + Customer Inquiry
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  Part 1: Return Risk ML   │   │ Part 2: TensorFlow Lite   │
│  Random Forest Pipeline   │   │ Product Image Classifier  │
│ (Threshold Tuned, joblib) │   │ (model_quant.tflite)      │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              ▼                               ▼
       Return Risk Tier               Detected Category
       & Probability                  & Confidence Score
              │                               │
              └───────────────┬───────────────┘
                              ▼
                ┌───────────────────────────┐
                │   Part 3: Support Agent   │
                │  Streamlit + Mistral AI   │
                └─────────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │   OrderSense-AI Web UI    │
                │     Streamlit Dashboard   │
                └───────────────────────────┘
```

---

## 🚀 Key Features

1. **Part 1 — Return Risk Prediction (ML Pipeline)**:
   - RandomForestClassifier trained with class-weight balancing and decision threshold tuning.
   - Evaluates order price, discount %, payment method, customer tenure, previous returns, and delivery timeline.
   - Outputs Return Probability and Risk Level (`LOW`, `MEDIUM`, `HIGH`).

2. **Part 2 — Product Image Classification (TensorFlow Lite)**:
   - Inference powered by Google Teachable Machine exported **TensorFlow Lite** model (`model_quant.tflite`).
   - Categorizes product photos into e-commerce categories (`Electronics`, `Apparel`, `Footwear`, `Home`, `Beauty`).

3. **Part 3 — Streamlit Web Dashboard & Mistral AI Agent**:
   - Single-command execution: `streamlit run app.py`.
   - Direct integration with the official **Mistral API** (`mistralai` SDK) and local fallback engine.
   - Interactive preset test scenarios (`High Risk`, `Medium Risk`, `Low Risk`).

---

## ⚙️ How to Run

### 1. Activate Virtual Environment
Open PowerShell inside the project directory:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Launch OrderSense-AI Dashboard
```powershell
streamlit run app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.
