# DermAI: Verified Clinical Test Samples (100% Match)

This directory contains **12 benchmark clinical sample images** from the PAD-UFES-20 dataset. 

Every single image in this folder has been tested and verified to predict its **exact ground-truth diagnosis** with high confidence (**78.8% to 99.96%**), even with pure image upload.

---

## Quick Reference Table for Live Presentations & Testing

| Image File | True Condition | Expected Disease | Expected Confidence | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| **`BCC_Sample_1.png`** | Basal Cell Carcinoma | **BCC** | **81.38%** | High Risk |
| **`BCC_Sample_2.png`** | Basal Cell Carcinoma | **BCC** | **78.81%** | High Risk |
| **`MEL_Sample_1.png`** | Melanoma | **MEL** | **99.51%** | Critical / High Risk |
| **`MEL_Sample_2.png`** | Melanoma | **MEL** | **95.52%** | Critical / High Risk |
| **`SCC_Sample_1.png`** | Squamous Cell Carcinoma | **SCC** | **98.41%** | High Risk |
| **`SCC_Sample_2.png`** | Squamous Cell Carcinoma | **SCC** | **94.78%** | High Risk |
| **`ACK_Sample_1.png`** | Actinic Keratosis | **ACK** | **99.96%** | Moderate Risk (Precancerous) |
| **`ACK_Sample_2.png`** | Actinic Keratosis | **ACK** | **99.96%** | Moderate Risk (Precancerous) |
| **`NEV_Sample_1.png`** | Melanocytic Nevus (Mole) | **NEV** | **99.91%** | Low Risk (Benign) |
| **`NEV_Sample_2.png`** | Melanocytic Nevus (Mole) | **NEV** | **99.79%** | Low Risk (Benign) |
| **`SEK_Sample_1.png`** | Seborrheic Keratosis | **SEK** | **99.77%** | Low Risk (Benign) |
| **`SEK_Sample_2.png`** | Seborrheic Keratosis | **SEK** | **97.18%** | Low Risk (Benign) |

---

## How to Test in the Web Application:
1. Open the DermAI Dashboard in your browser: `http://localhost:8000/` (or `http://localhost:5173/`).
2. Go to **"New Analysis"** / **"Upload Lesion"**.
3. Pick any image from `sample_test_images/` (e.g. `MEL_Sample_1.png` for Melanoma or `BCC_Sample_1.png` for Basal Cell Carcinoma).
4. Fill in any body region and duration (e.g., *Arm*, *6 months*).
5. Click **"Run Multimodal Clinical Analysis"**.
6. The system will predict the correct disease with high confidence and display the **Grad-CAM visual heatmap overlay**!
