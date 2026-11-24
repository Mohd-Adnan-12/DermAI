# DermAI — Multimodal Deep Learning Platform for Dermatological Screening & Clinical Triage

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ROC-AUC](https://img.shields.io/badge/Macro%20ROC--AUC-90.31%25-brightgreen)](https://github.com/Mohd-Adnan-12/DermAI)
[![Melanoma AUC](https://img.shields.io/badge/Melanoma%20AUC-97.69%25-red)](https://github.com/Mohd-Adnan-12/DermAI)

**DermAI** is an end-to-end, clinical-grade medical decision-support platform designed to assist in early detection, risk stratification, and automated triage of skin lesions. 

Trained on the **PAD-UFES-20** clinical smartphone dataset, DermAI leverages **Multimodal Deep Learning** by fusing high-resolution dermoscopic image representations (EfficientNet-B0) with 64-dimensional clinical patient metadata (age, biological gender, lesion anatomical region, itch, bleeding, elevation, growth rate).

---

## 🌟 Key Highlights & Innovations

- 🧠 **Multimodal Neural Fusion:** Combines deep convolutional vision embeddings with tabular clinical epidemiology features for diagnostic accuracy surpassing image-only CNNs.
- 🔬 **Explainable AI (Grad-CAM):** Generates dual-panel visual attention heatmaps and overlays showing exactly which morphological tissue regions triggered the model's prediction.
- ✉️ **Automated Clinical Email Alerts:** Asynchronously dispatches medical-grade HTML screening alerts and triage pathways directly from `dermai.diagnostics@gmail.com` to the registered patient's email.
- 🚨 **Actionable Risk Stratification:** Automatically categorizes lesions into **Critical (Urgent Surgical Biopsy)**, **High**, **Moderate**, and **Low Risk (Routine Self-Monitoring)**.
- 🩺 **Clinician Diagnostic Hub:** Dedicated role-based access control for licensed dermatologists to review, audit, verify AI predictions, and append verified clinical notes.
- 📥 **One-Click Diagnostic Export:** Generates downloadable clinical diagnostic summaries in **PDF**, **CSV**, and **JSON** formats.

---

## 📊 Neural Model Performance Benchmark

The model was rigorously trained, validated, and tested on independent patient partitions of the **PAD-UFES-20** dataset:

| Metric | Score | Clinical Scope |
| :--- | :---: | :--- |
| **Training Accuracy** | **98.07%** | 2,298 Total Cases |
| **Validation Accuracy** | **77.55%** | Cross-entropy loss minimization |
| **Test Accuracy** | **76.74%** | 460 Unseen Test Cases |
| **Macro ROC-AUC** | **90.31%** | Multi-Class One-vs-Rest (OvR) |
| **Melanoma (`MEL`) ROC-AUC** | **97.69%** | High sensitivity for malignant skin cancer |

### Diagnostic Conditions Screened (6 Classes)
1. **MEL (Melanoma):** High-risk malignant cancer of melanocytes *(ROC-AUC: 97.69%)*.
2. **BCC (Basal Cell Carcinoma):** Locally invasive non-melanoma skin cancer *(ROC-AUC: 89.98%)*.
3. **SCC (Squamous Cell Carcinoma):** Invasive keratinocyte carcinoma *(ROC-AUC: 94.39%)*.
4. **ACK (Actinic Keratosis):** Precancerous dysplastic sun-damaged lesion *(ROC-AUC: 89.24%)*.
5. **NEV (Melanocytic Nevus):** Benign proliferation of pigment cells *(ROC-AUC: 88.08%)*.
6. **SEK (Seborrheic Keratosis):** Common benign keratinocytic growth *(ROC-AUC: 82.49%)*.

---

## 🛠️ System Architecture & Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                       React 19 + Vite                       │
│  (Edge-to-Edge Full-Screen UI, Grad-CAM Viewer, Reports)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTPS / JSON REST
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Engine                   │
│   (JWT Auth, Quality Assessment, Image Preprocessing)       │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼─────────────┐ ┌──────────────▼───────────────┐
│     PyTorch Multimodal     │ │       PostgreSQL Database     │
│   (EfficientNet-B0 +       │ │ (Users, Profiles, Cases,     │
│    Metadata Neural Fusion) │ │  Images, Predictions History) │
└──────────────┬─────────────┘ └──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│       Asynchronous Email Dispatch Engine (SMTP Relay)       │
│  (Dispatches clinical HTML reports from dedicated sender)   │
└─────────────────────────────────────────────────────────────┘
```

- **Frontend:** React 19, Vite, Modern CSS3 Flexbox/Grid, Canvas API.
- **Backend:** FastAPI, Python 3.11, Uvicorn, Pydantic v2, Python-Jose (JWT), Pwdlib (Argon2).
- **Machine Learning:** PyTorch, Torchvision, Scikit-Learn, NumPy, Pandas, OpenCV, Pillow.
- **Database:** PostgreSQL 16 (Psycopg 3).
- **Notifications:** Python standard `smtplib` + `email.mime` via FastAPI `BackgroundTasks`.
- **Containerization:** Multi-stage `Dockerfile` (Node 20 builder + Python 3.11 slim runtime).

---

## 🚀 Quick Start & Local Development

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL

### 1. Clone & Set Up Python Virtual Environment
```bash
git clone https://github.com/Mohd-Adnan-12/DermAI.git
cd DermAI

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your database URL and email settings in `.env`:
```ini
DERMAI_JWT_SECRET=your-random-jwt-secret
DATABASE_URL=postgresql:///dermai_db

# Automated Email Notification Settings (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_project_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password
SMTP_FROM_EMAIL=your_project_email@gmail.com
SMTP_USE_TLS=true
```

### 3. Start Backend & Frontend
```bash
# Terminal 1: Start FastAPI Backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start Vite Frontend
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 🐳 Docker Production Deployment

Run the complete multi-container stack (PostgreSQL + FastAPI + React static bundle) with a single command:

```bash
docker compose up --build -d
```
Access the application at **`http://localhost:8000`**.

---

## ☁️ Cloud Deployment (Render / Railway)

1. **Database:** Create a free PostgreSQL instance on [Render.com](https://render.com).
2. **Web Service:** Connect your GitHub repository `Mohd-Adnan-12/DermAI` using the **Docker** runtime.
3. **Environment Variables:**
   - `DATABASE_URL` = `<your-render-postgresql-url>`
   - `DERMAI_JWT_SECRET` = `dermai-secret-key-production-change-this-9e8c45928a3f4e`
   - `SMTP_HOST` = `smtp.gmail.com`
   - `SMTP_PORT` = `587`
   - `SMTP_USER` = `dermai.diagnostics@gmail.com`
   - `SMTP_PASSWORD` = `<your-google-app-password>`
   - `SMTP_FROM_EMAIL` = `dermai.diagnostics@gmail.com`
   - `SMTP_USE_TLS` = `true`
4. Click **Deploy** to receive your live HTTPS domain.

---

## ⚠️ Mandatory Medical Decision-Support Notice

> **IMPORTANT CLINICAL DISCLAIMER:**
> DermAI is an artificial intelligence-assisted clinical decision-support tool developed for research, preliminary screening, and triaging prioritization. It is **NOT** an autonomous diagnostic medical device or a substitute for professional clinical judgment. All preliminary findings should be verified through in-person clinical examination, dermoscopy, and histopathological biopsy by a licensed dermatologist or qualified healthcare provider.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
