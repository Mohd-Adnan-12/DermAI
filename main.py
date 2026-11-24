
from pathlib import Path
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timedelta, timezone
import os
import re
import json
import base64

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import psycopg
import jwt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import joblib
import pickle
import pandas as pd

from pydantic import BaseModel, Field
from pwdlib import PasswordHash
from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Depends,
    Form,
    BackgroundTasks
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ---------------- APPLICATION ----------------

app = FastAPI(
    title="DermAI API",
    description="API for preliminary AI-assisted skin image analysis",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- CONFIGURATION ----------------

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///dermai_db")

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

# Trained checkpoints & artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parent / "DermAI_Artifacts"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "DermAI_Final_Model.pth"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "DermAI_Metadata_Preprocessor.pkl"
CLASS_INFO_PATH = ARTIFACTS_DIR / "DermAI_Class_Information.pkl"
STANDALONE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "dermai_efficientnet_b0_for_api.pth"

DEVICE = torch.device("cpu")  # CPU default for API inference stability
classifier_model = None
classifier_classes = []
is_multimodal = False
metadata_preprocessor = None
class_info = {}

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

DISEASE_METADATA = {
    "ACK": {
        "name": "Actinic Keratosis",
        "category": "Precancerous Lesion",
        "description": "Rough, scaly patch caused by chronic sun exposure; potential precursor to squamous cell carcinoma.",
        "risk_level": "Moderate Risk",
        "severity": "Moderate (Precancerous)",
        "treatment_required": "Yes — Treatment Advised",
        "treatment_type": "Cryotherapy (Liquid Nitrogen), Topical 5-Fluorouracil (5-FU), or Photodynamic Therapy (PDT)",
        "recommendation": "Consult a dermatologist for clinical evaluation and targeted precancerous lesion treatment."
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "description": "The most common form of skin cancer. Locally invasive with high cure rates when excised early.",
        "risk_level": "High Risk",
        "severity": "High (Malignant Neoplasm)",
        "treatment_required": "Yes — Prompt Surgical Treatment Required",
        "treatment_type": "Mohs Micrographic Surgery, Standard Surgical Excision, or Curettage & Electrodesiccation",
        "recommendation": "Prompt evaluation by a dermatologist or dermatologic surgeon for diagnostic biopsy and surgical removal."
    },
    "MEL": {
        "name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "description": "The most serious and aggressive type of skin cancer, arising from pigment-producing melanocytes.",
        "risk_level": "High Risk",
        "severity": "Critical (High-Risk Malignancy)",
        "treatment_required": "Yes — Urgent Treatment Required",
        "treatment_type": "Urgent Wide Local Excision, Sentinel Lymph Node Biopsy, and Comprehensive Oncological Staging",
        "recommendation": "Urgent dermatological assessment and surgical excision biopsy strongly advised without delay."
    },
    "NEV": {
        "name": "Melanocytic Nevus",
        "category": "Benign Mole",
        "description": "Common benign proliferation of melanocytes. Non-cancerous and typically harmless.",
        "risk_level": "Low Risk",
        "severity": "Low (Benign)",
        "treatment_required": "No Immediate Treatment Required",
        "treatment_type": "Routine Self-Monitoring (ABCDE: Asymmetry, Border, Color, Diameter, Evolving)",
        "recommendation": "Benign finding. No medical intervention required. Periodically inspect for changes in shape or color."
    },
    "SCC": {
        "name": "Squamous Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "description": "Second most common skin cancer. Arises from epidermal keratinocytes and can metastasize if neglected.",
        "risk_level": "High Risk",
        "severity": "High (Malignant Neoplasm)",
        "treatment_required": "Yes — Urgent Treatment Required",
        "treatment_type": "Surgical Excision with Clear Margins, Mohs Surgery, or Radiation Therapy",
        "recommendation": "Prompt clinical evaluation and histopathological biopsy recommended to prevent deep tissue invasion."
    },
    "SEK": {
        "name": "Seborrheic Keratosis",
        "category": "Benign Keratosis",
        "description": "Common non-cancerous epidermal growth that typically looks waxy, scaly, or 'stuck on'.",
        "risk_level": "Low Risk",
        "severity": "Low (Benign)",
        "treatment_required": "No Treatment Required",
        "treatment_type": "Cosmetic Removal Only (if physically irritated by clothing or cosmetically desired)",
        "recommendation": "Benign condition. No treatment required unless irritated, bleeding, or causing discomfort."
    }
}


class DermAIMultimodalModel(nn.Module):
    def __init__(self, num_metadata_features=85, num_classes=6):
        super().__init__()
        self.image_model = models.efficientnet_b0(weights=None)
        self.image_model.classifier = nn.Identity()
        self.image_branch = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.metadata_branch = nn.Sequential(
            nn.Linear(num_metadata_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, metadata):
        image_features = self.image_model(images)
        image_features = self.image_branch(image_features)
        metadata_features = self.metadata_branch(metadata)
        combined_features = torch.cat([image_features, metadata_features], dim=1)
        return self.classifier(combined_features)


def load_classifier():
    """Load the trained DermAI multimodal or standalone EfficientNet-B0 checkpoint."""
    global classifier_model, classifier_classes, is_multimodal, metadata_preprocessor, class_info

    # 1. Try loading multimodal model from DermAI_Artifacts
    if FINAL_MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file() and CLASS_INFO_PATH.is_file():
        try:
            with open(CLASS_INFO_PATH, "rb") as f:
                class_info = pickle.load(f)
            metadata_preprocessor = joblib.load(PREPROCESSOR_PATH)
            checkpoint = torch.load(FINAL_MODEL_PATH, map_location=DEVICE, weights_only=False)

            classifier_classes = [str(c) for c in checkpoint.get("class_names", class_info.get("class_names", []))]
            num_meta = int(checkpoint.get("metadata_feature_count", 85))

            model = DermAIMultimodalModel(num_metadata_features=num_meta, num_classes=len(classifier_classes))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(DEVICE)
            model.eval()

            classifier_model = model
            is_multimodal = True
            print(
                "DermAI Multimodal Classifier loaded successfully from DermAI_Artifacts!",
                f"| classes: {classifier_classes}",
                f"| test macro F1: {checkpoint.get('test_macro_f1', 'N/A')}"
            )
            return
        except Exception as exc:
            print(f"Warning: Could not load multimodal model ({exc}). Trying fallback...")

    # 2. Fallback to standalone image-only model
    if STANDALONE_MODEL_PATH.is_file():
        checkpoint = torch.load(STANDALONE_MODEL_PATH, map_location=DEVICE, weights_only=False)
        classifier_classes = [str(label) for label in checkpoint["class_names"]]

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classifier_classes))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(DEVICE)
        model.eval()

        classifier_model = model
        is_multimodal = False
        print("DermAI standalone classifier loaded:", checkpoint.get("model_name", "EfficientNet-B0"), "| classes:", classifier_classes)
        return

    print("WARNING: No DermAI model checkpoints found in DermAI_Artifacts or models folder.")
    classifier_model = None
    classifier_classes = []


# ---------------- EMAIL NOTIFICATION CONFIGURATION (SMTP) ----------------
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", "").strip() or SMTP_USER or "alerts@dermai.clinical"
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

# HTTPS Email APIs for Cloud Providers (Render Free Tier blocks raw SMTP port 587)
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "").strip()
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "").strip() or SMTP_FROM_EMAIL or "dermai.diagnostics@gmail.com"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()


def send_via_https_api(recipient_email: str, patient_name: str, subject: str, html_content: str, text_content: str) -> bool:
    """Send email over standard HTTPS Port 443 to bypass cloud provider SMTP blocks."""
    import urllib.request
    import urllib.error

    if BREVO_API_KEY:
        try:
            req_data = json.dumps({
                "sender": {"name": "DermAI Clinical Hub", "email": BREVO_SENDER_EMAIL},
                "to": [{"email": recipient_email, "name": patient_name}],
                "subject": subject,
                "htmlContent": html_content,
                "textContent": text_content
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.brevo.com/v3/smtp/email",
                data=req_data,
                headers={
                    "api-key": BREVO_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status in (200, 201, 202):
                    print(f"📧 [EMAIL SENT via Brevo HTTPS API] Live notification dispatched to {recipient_email}")
                    return True
        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode("utf-8", errors="ignore")
            print(f"⚠️ [Brevo HTTPS API HTTP {http_err.code}]: {error_body}")
        except Exception as exc:
            print(f"⚠️ [Brevo HTTPS API Error]: {exc}")

    if RESEND_API_KEY:
        try:
            req_data = json.dumps({
                "from": "DermAI Clinical Hub <onboarding@resend.dev>",
                "to": [recipient_email],
                "subject": subject,
                "html": html_content,
                "text": text_content
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.resend.com/emails",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 201, 202):
                    print(f"📧 [EMAIL SENT via Resend HTTPS API] Live notification dispatched to {recipient_email}")
                    return True
        except Exception as exc:
            print(f"⚠️ [Resend HTTPS API Error]: {exc}")

    return False


def build_report_email_html(patient_name: str, case_id: int, prediction: dict, affected_body_part: str = "Unspecified", condition_duration: str = "Unspecified") -> str:
    disease_name = prediction.get("disease_name") or prediction.get("predicted_class", "Undetermined Condition")
    code = prediction.get("predicted_class", "N/A")
    confidence = prediction.get("confidence", 0.0)
    risk_level = prediction.get("risk_level", "Moderate Risk")
    recommendation = prediction.get("clinical_recommendation") or prediction.get("recommendation", "Clinical consultation advised.")
    treatment_req = prediction.get("treatment_required", "Clinical Assessment Advised")

    badge_bg = "#dc2626" if "high" in risk_level.lower() else ("#d97706" if "mod" in risk_level.lower() else "#16a34a")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DermAI Clinical Diagnostic Alert</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; color: #1e293b; margin: 0; padding: 24px 12px;">
  <table role="presentation" width="100%" border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" style="max-width: 620px; background-color: #ffffff; border-radius: 14px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.08);" border="0" cellspacing="0" cellpadding="0">
          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #0b554d 0%, #148074 100%); padding: 28px 24px; text-align: left; color: #ffffff;">
              <div style="display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.25); border-radius: 9999px; padding: 4px 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
                Automated Clinical Notification
              </div>
              <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px;">
                DermAI Diagnostic Screening Alert
              </h1>
              <p style="margin: 4px 0 0 0; font-size: 13px; color: #d1fae5;">
                Multimodal Deep Learning Dermatological Triage
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding: 28px 24px;">
              <table role="presentation" width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-bottom: 16px;">
                <tr>
                  <td>
                    <span style="display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700;">
                      Case Reference #{case_id}
                    </span>
                  </td>
                  <td align="right" style="font-size: 12px; color: #64748b;">
                    {datetime.now(timezone.utc).strftime('%b %d, %Y · %H:%M UTC')}
                  </td>
                </tr>
              </table>

              <p style="margin: 0 0 14px 0; font-size: 15px; line-height: 1.5; color: #1e293b;">
                Dear <strong>{patient_name}</strong>,
              </p>
              <p style="margin: 0 0 20px 0; font-size: 14px; line-height: 1.5; color: #475569;">
                Your skin lesion image and clinical case information have been analyzed by the DermAI Multimodal Neural Screening Engine. Here is your automated preliminary summary:
              </p>

              <!-- Diagnostic Result Box -->
              <table role="presentation" width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; margin-bottom: 22px; overflow: hidden;">
                <tr>
                  <td style="padding: 18px 20px;">
                    <table role="presentation" width="100%" border="0" cellspacing="0" cellpadding="0">
                      <tr>
                        <td>
                          <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                            Preliminary Detection
                          </div>
                          <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">
                            {disease_name} <span style="font-size: 13px; color: #64748b; font-weight: 600;">({code})</span>
                          </div>
                        </td>
                        <td align="right">
                          <span style="display: inline-block; background-color: {badge_bg}; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 700; text-transform: uppercase;">
                            {risk_level}
                          </span>
                        </td>
                      </tr>
                    </table>

                    <div style="margin-top: 14px; padding-top: 14px; border-top: 1px solid #e2e8f0; font-size: 13.5px; color: #334155;">
                      <div><strong>Confidence Score:</strong> {confidence}%</div>
                      <div style="margin-top: 6px;"><strong>Clinical Triage Protocol:</strong> {treatment_req}</div>
                      <div style="margin-top: 6px;"><strong>Action Pathway:</strong> {recommendation}</div>
                    </div>
                  </td>
                </tr>
              </table>

              <!-- Case Metadata Table -->
              <table role="presentation" width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-bottom: 22px; font-size: 13px; border-collapse: collapse;">
                <tr>
                  <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b; font-weight: 600; width: 38%;">Affected Anatomical Region:</td>
                  <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #0f172a; font-weight: 500;">{affected_body_part}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b; font-weight: 600;">Reported Duration:</td>
                  <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #0f172a; font-weight: 500;">{condition_duration}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b; font-weight: 600;">Neural Architecture:</td>
                  <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #0f172a; font-weight: 500;">EfficientNet-B0 + 64-Dim Clinical Feature Fusion</td>
                </tr>
                <tr>
                  <td style="padding: 8px 10px; color: #64748b; font-weight: 600;">Explainability:</td>
                  <td style="padding: 8px 10px; color: #0f172a; font-weight: 500;">Grad-CAM Lesion Heatmap Available in Portal</td>
                </tr>
              </table>

              <!-- Decision Support Notice -->
              <div style="background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 14px 16px; border-radius: 0 8px 8px 0; font-size: 11.5px; color: #92400e; line-height: 1.55;">
                <strong>⚠️ MANDATORY CLINICAL DECISION-SUPPORT NOTICE:</strong><br>
                This automated report alert is generated by an artificial intelligence decision-support model to assist pre-screening prioritization. It is <strong>NOT</strong> an autonomous medical diagnosis. Definitive diagnosis requires in-person examination, dermoscopy, and histopathological biopsy by a licensed physician or dermatologist.
              </div>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color: #f8fafc; padding: 18px 24px; text-align: center; border-top: 1px solid #e2e8f0; font-size: 11px; color: #94a3b8;">
              DermAI Clinical Technologies · PAD-UFES-20 Calibrated · All rights reserved.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def build_report_email_text(patient_name: str, case_id: int, prediction: dict, affected_body_part: str = "Unspecified") -> str:
    disease_name = prediction.get("disease_name") or prediction.get("predicted_class", "Undetermined")
    confidence = prediction.get("confidence", 0.0)
    risk_level = prediction.get("risk_level", "Moderate Risk")
    recommendation = prediction.get("clinical_recommendation") or prediction.get("recommendation", "Clinical consultation advised.")

    return f"""DermAI Clinical Screening Alert - Case #{case_id}
==========================================================
Dear {patient_name},

Your skin lesion analysis has been processed by the DermAI Multimodal Neural Diagnostic Pipeline:

- Case Reference: #{case_id}
- Preliminary Finding: {disease_name} ({prediction.get('predicted_class', 'N/A')})
- Risk Stratification: {risk_level}
- Model Confidence: {confidence}%
- Affected Region: {affected_body_part}
- Clinical Recommendation: {recommendation}

NOTICE: This automated screening alert does NOT constitute a definitive medical diagnosis. Please consult a certified dermatologist for professional clinical evaluation.

DermAI Clinical Decision-Support Hub
"""


def build_clinician_review_email_html(patient_name: str, case_id: int, clinician_name: str, clinical_notes: str, status: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; color: #1e293b; padding: 24px;">
  <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 28px;">
    <h2 style="color: #0b554d; margin-top: 0;">🩺 Clinician Case Review Completed</h2>
    <p>Dear <strong>{patient_name}</strong>,</p>
    <p>Your case <strong>#{case_id}</strong> has been reviewed by a licensed clinician on the DermAI platform.</p>
    
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px; margin: 20px 0;">
      <p style="margin: 0 0 8px 0;"><strong>Reviewing Clinician:</strong> Dr. {clinician_name}</p>
      <p style="margin: 0 0 8px 0;"><strong>Case Status:</strong> <span style="text-transform: capitalize; font-weight: 700; color: #047857;">{status}</span></p>
      <p style="margin: 0;"><strong>Clinical Assessment Notes:</strong></p>
      <p style="margin: 4px 0 0 0; color: #334155; font-style: italic;">"{clinical_notes or 'Case verified and approved by clinical review.'}"</p>
    </div>

    <p style="font-size: 13px; color: #64748b;">You can log into your DermAI portal to view the complete diagnostic record and download your updated clinical report.</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
    <p style="font-size: 11px; color: #94a3b8; text-align: center;">DermAI Clinical Platform &middot; Confidential Medical Correspondence</p>
  </div>
</body>
</html>"""


def send_analysis_email_alert(
    recipient_email: str,
    patient_name: str,
    case_id: int,
    prediction: dict,
    affected_body_part: str = "Unspecified",
    condition_duration: str = "Unspecified"
) -> bool:
    if not recipient_email or "@" not in recipient_email:
        print(f"[EMAIL] Skipped alert dispatch: Invalid recipient '{recipient_email}'")
        return False

    subject = f"🩺 DermAI Diagnostic Alert: Case #{case_id} Preliminary Screening Result"
    html_content = build_report_email_html(patient_name, case_id, prediction, affected_body_part, condition_duration)
    text_content = build_report_email_text(patient_name, case_id, prediction, affected_body_part)

    # Try HTTPS API first if configured (Port 443 - bypasses Render Free Tier SMTP block)
    if BREVO_API_KEY or RESEND_API_KEY:
        if send_via_https_api(recipient_email, patient_name, subject, html_content, text_content):
            return True

    if not SMTP_HOST:
        print(f"\n=======================================================")
        print(f"📧 [EMAIL DISPATCH SIMULATION] Report Alert Dispatched")
        print(f"-------------------------------------------------------")
        print(f"To:          {recipient_email} ({patient_name})")
        print(f"Subject:     {subject}")
        print(f"Case ID:     #{case_id}")
        print(f"Finding:     {prediction.get('disease_name')} ({prediction.get('predicted_class')})")
        print(f"Confidence:  {prediction.get('confidence')}% | Risk: {prediction.get('risk_level')}")
        print(f"Status:      Delivered to {recipient_email} (Local Simulation Mode)")
        print(f"Note:        Configure SMTP_HOST in .env to relay through live mail server")
        print(f"=======================================================\n")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"DermAI Clinical Hub <{SMTP_FROM_EMAIL}>"
        msg["To"] = recipient_email
        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        if SMTP_USE_TLS:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
                server.starttls()
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM_EMAIL, [recipient_email], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM_EMAIL, [recipient_email], msg.as_string())

        print(f"📧 [EMAIL SENT] Live report notification dispatched to {recipient_email}")
        return True
    except Exception as exc:
        print(f"⚠️ [EMAIL FAILED] Could not send live email to {recipient_email}: {exc}")
        return False


def send_clinician_review_email_alert(
    recipient_email: str,
    patient_name: str,
    case_id: int,
    clinician_name: str,
    clinical_notes: str,
    status: str
) -> bool:
    if not recipient_email or "@" not in recipient_email:
        return False

    subject = f"🩺 DermAI Case Update: Clinician Review for Case #{case_id}"
    html_content = build_clinician_review_email_html(patient_name, case_id, clinician_name, clinical_notes, status)
    text_content = f"Case #{case_id} has been reviewed by Dr. {clinician_name}. Status: {status}. Notes: {clinical_notes}"

    # Try HTTPS API first if configured (Port 443 - bypasses Render Free Tier SMTP block)
    if BREVO_API_KEY or RESEND_API_KEY:
        if send_via_https_api(recipient_email, patient_name, subject, html_content, text_content):
            return True

    if not SMTP_HOST:
        print(f"\n📧 [EMAIL DISPATCH SIMULATION] Clinician Review Alert Dispatched to {recipient_email} (Case #{case_id})\n")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"DermAI Clinical Hub <{SMTP_FROM_EMAIL}>"
        msg["To"] = recipient_email
        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
            if SMTP_USE_TLS:
                server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, [recipient_email], msg.as_string())
        print(f"📧 [EMAIL SENT] Clinician review alert sent to {recipient_email}")
        return True
    except Exception as exc:
        print(f"⚠️ [EMAIL FAILED] Could not send email: {exc}")
        return False


def generate_gradcam_overlay(image_tensor, metadata_tensor, original_pil_image, target_class_idx: int):
    """Generate Grad-CAM heatmap overlay for the predicted disease class."""
    if not is_multimodal or classifier_model is None:
        return None

    try:
        activations = None

        def hook_fn(module, inputs, output):
            nonlocal activations
            activations = output
            output.retain_grad()

        target_layer = classifier_model.image_model.features[-1]
        handle = target_layer.register_forward_hook(hook_fn)

        classifier_model.eval()
        classifier_model.zero_grad()

        img_input = image_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            outputs = classifier_model(img_input, metadata_tensor)
            score = outputs[0, target_class_idx]
            score.backward()

        handle.remove()

        if activations is None or activations.grad is None:
            return None

        grads = activations.grad.detach()
        acts = activations.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        orig_resized = np.array(original_pil_image.resize((224, 224)))
        if len(orig_resized.shape) == 2:
            orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)
        elif orig_resized.shape[2] == 4:
            orig_resized = orig_resized[:, :, :3]

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_resized, 0.6, heatmap, 0.4, 0)

        # Raw Heatmap
        heatmap_pil = Image.fromarray(heatmap)
        buffered_heat = BytesIO()
        heatmap_pil.save(buffered_heat, format="JPEG", quality=90)
        heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered_heat.getvalue()).decode("utf-8")

        # Overlay Heatmap
        overlay_pil = Image.fromarray(overlay)
        buffered_ov = BytesIO()
        overlay_pil.save(buffered_ov, format="JPEG", quality=90)
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered_ov.getvalue()).decode("utf-8")

        return overlay_b64, heatmap_b64
    except Exception as exc:
        print(f"Grad-CAM generation error: {exc}")
        return None, None


def predict_skin_disease(image_bytes: bytes, clinical_dict: dict = None) -> dict:
    """Classify skin image using the trained model and clinical metadata, with Grad-CAM explainability."""
    if classifier_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The DermAI classification model is not loaded. "
                "Ensure DermAI_Artifacts or models checkpoints are present."
            )
        )

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            rgb_image = source.convert("RGB")
            input_tensor = classifier_transform(rgb_image).unsqueeze(0).to(DEVICE)

        # Prepare metadata tensor
        if is_multimodal and metadata_preprocessor is not None:
            clinical_features = class_info.get("clinical_features", [])
            categorical_features = class_info.get("categorical_features", [])
            meta_row = {col: np.nan for col in clinical_features}
            if clinical_dict:
                for k, v in clinical_dict.items():
                    if k in meta_row and v is not None:
                        meta_row[k] = v
            meta_df = pd.DataFrame([meta_row])
            for col in categorical_features:
                if col in meta_df.columns:
                    meta_df[col] = meta_df[col].map(lambda x: str(x) if pd.notna(x) else np.nan)
            meta_vec = metadata_preprocessor.transform(meta_df)
            meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).to(DEVICE)
        else:
            meta_tensor = torch.zeros((1, 85), dtype=torch.float32).to(DEVICE)

        with torch.inference_mode():
            if is_multimodal:
                logits = classifier_model(input_tensor, meta_tensor)
            else:
                logits = classifier_model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        confidence, predicted_index = torch.max(probabilities, dim=0)
        pred_idx_int = int(predicted_index.item())
        pred_class = classifier_classes[pred_idx_int]
        conf_float = round(float(confidence.item()) * 100, 2)

        probability_map = {
            classifier_classes[index]: round(float(probabilities[index].item()) * 100, 2)
            for index in range(len(classifier_classes))
        }

        info = DISEASE_METADATA.get(pred_class, {
            "name": pred_class,
            "category": "Skin Condition",
            "description": "Dermatological finding.",
            "risk_level": "Moderate Risk",
            "recommendation": "Consult a healthcare provider for professional evaluation."
        })

        risk_level = info["risk_level"]
        if conf_float < 55.0:
            risk_level = "Requires Clinical Review"

        # Generate Grad-CAM explainability heatmap & overlay
        gradcam_overlay_b64, gradcam_raw_b64 = None, None
        if is_multimodal:
            gradcam_overlay_b64, gradcam_raw_b64 = generate_gradcam_overlay(input_tensor, meta_tensor, rgb_image, pred_idx_int)

        treatment_req = info.get("treatment_required", "Consult Physician")
        treatment_proc = info.get("treatment_type", "Clinical Evaluation")
        if conf_float < 55.0:
            treatment_req = "In-Person Clinical Evaluation Required"
            treatment_proc = "Diagnostic Dermoscopy and Biopsy Assessment"

        return {
            "model_name": "EfficientNet-B0 + Clinical Metadata Fusion" if is_multimodal else "EfficientNet-B0 Balanced",
            "predicted_class": pred_class,
            "disease_name": info["name"],
            "disease_category": info["category"],
            "disease_description": info["description"],
            "risk_level": risk_level,
            "severity": info.get("severity", "Moderate"),
            "treatment_required": treatment_req,
            "treatment_type": treatment_proc,
            "clinical_recommendation": info["recommendation"],
            "confidence": conf_float,
            "class_probabilities": probability_map,
            "gradcam_image": gradcam_overlay_b64,
            "gradcam_raw_heatmap": gradcam_raw_b64,
            "input_size": "224x224"
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="The image could not be processed by the classification model."
        ) from exc


# Load once when the API module starts. The API can still start without the
# checkpoint, but classification requests will return HTTP 503 until it is added.
load_classifier()

JWT_SECRET = os.environ.get("DERMAI_JWT_SECRET")

if not JWT_SECRET:
    raise RuntimeError(
        "DERMAI_JWT_SECRET environment variable is not set."
    )

JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

password_hash = PasswordHash.recommended()
security = HTTPBearer()


# ---------------- DATABASE ----------------

def get_connection():
    return psycopg.connect(DATABASE_URL)


def init_db():
    """Ensure database tables and columns exist on startup for zero-friction cloud deployment."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(150) UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role VARCHAR(30) NOT NULL DEFAULT 'user',
                        is_approved BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS patient_profiles (
                        patient_id SERIAL PRIMARY KEY,
                        user_id INTEGER UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        patient_name VARCHAR(100) NOT NULL,
                        age INTEGER NOT NULL CHECK (age BETWEEN 1 AND 120),
                        gender VARCHAR(30),
                        weight_kg NUMERIC(5,2),
                        medical_history TEXT,
                        previous_skin_treatments TEXT,
                        current_medications TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cases (
                        case_id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(user_id),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(30) DEFAULT 'pending',
                        affected_body_part VARCHAR(100),
                        condition_duration VARCHAR(100),
                        symptoms JSONB DEFAULT '[]'::jsonb,
                        clinical_notes TEXT,
                        reviewed_by INTEGER REFERENCES users(user_id),
                        reviewed_at TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        image_id SERIAL PRIMARY KEY,
                        case_id INTEGER NOT NULL REFERENCES cases(case_id),
                        image_path TEXT NOT NULL,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        prediction_id SERIAL PRIMARY KEY,
                        image_id INTEGER NOT NULL REFERENCES images(image_id),
                        predicted_class VARCHAR(10) NOT NULL,
                        confidence NUMERIC(5,4),
                        model_version VARCHAR(50),
                        risk_level VARCHAR(50),
                        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        history_id SERIAL PRIMARY KEY,
                        prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
                        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_approved BOOLEAN NOT NULL DEFAULT TRUE;")
                cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS affected_body_part VARCHAR(100);")
                cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS condition_duration VARCHAR(100);")
                cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS symptoms JSONB DEFAULT '[]'::jsonb;")
                cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS clinical_notes TEXT;")
                cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS reviewed_by INTEGER REFERENCES users(user_id);")
                cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;")
                cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS risk_level VARCHAR(50);")

                # Auto-seed default administrator account on fresh cloud database if no admin exists
                cur.execute("SELECT user_id FROM users WHERE role = 'admin' LIMIT 1")
                if not cur.fetchone():
                    admin_email = os.environ.get("ADMIN_EMAIL", "admin@dermai.com").strip().lower()
                    admin_pass = os.environ.get("ADMIN_PASSWORD", "Admin@123")
                    admin_hash = password_hash.hash(admin_pass)
                    cur.execute("""
                        INSERT INTO users (name, email, password_hash, role, is_approved)
                        VALUES (%s, %s, %s, 'admin', TRUE)
                        ON CONFLICT (email) DO UPDATE SET role = 'admin', is_approved = TRUE
                    """, ("DermAI Administrator", admin_email, admin_hash))
                    print(f"Default administrator account initialized ({admin_email} / {admin_pass}).")

                conn.commit()
                print("DermAI PostgreSQL schema verified and ready.")
    except Exception as exc:
        print(f"Database schema initialization warning (will retry on requests): {exc}")


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------- IMAGE QUALITY ASSESSMENT ----------------

# Quality checks use two levels:
# - Severe problems block submission.
# - Borderline quality produces a warning but allows submission.
# These are heuristic checks, not a medical assessment of the image.
MIN_IMAGE_WIDTH = 128
MIN_IMAGE_HEIGHT = 128
MIN_SHARPNESS = 8.0
MIN_BRIGHTNESS = 15.0
MAX_BRIGHTNESS = 245.0
MIN_CONTRAST = 8.0

# Values below these softer limits trigger a warning only.
RECOMMENDED_IMAGE_WIDTH = 256
RECOMMENDED_IMAGE_HEIGHT = 256
RECOMMENDED_SHARPNESS = 50.0
RECOMMENDED_MIN_BRIGHTNESS = 35.0
RECOMMENDED_MAX_BRIGHTNESS = 220.0
RECOMMENDED_CONTRAST = 20.0


def assess_image_quality(image_bytes: bytes) -> dict:
    """Check image dimensions, blur, brightness and contrast."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as an image."
        )

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    blocking_issues = []
    warnings = []
    suggestions = []

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        blocking_issues.append("Image resolution is too low to assess reliably.")
        suggestions.append(
            f"Choose an image at least {MIN_IMAGE_WIDTH} × {MIN_IMAGE_HEIGHT} pixels."
        )
    elif width < RECOMMENDED_IMAGE_WIDTH or height < RECOMMENDED_IMAGE_HEIGHT:
        warnings.append("Image resolution is below the recommended size.")
        suggestions.append("A larger, closer image may provide better results.")

    if sharpness < MIN_SHARPNESS:
        blocking_issues.append("Image is extremely blurry or out of focus.")
        suggestions.append("Hold the camera steady and retake the photo in focus.")
    elif sharpness < RECOMMENDED_SHARPNESS:
        warnings.append("Image may be slightly blurry.")
        suggestions.append("For better results, use a sharper, focused image.")

    if brightness < MIN_BRIGHTNESS:
        blocking_issues.append("Image is too dark to assess reliably.")
        suggestions.append("Move to a well-lit area and avoid shadows.")
    elif brightness < RECOMMENDED_MIN_BRIGHTNESS:
        warnings.append("Image appears somewhat dark.")
        suggestions.append("More even lighting may improve image quality.")
    elif brightness > MAX_BRIGHTNESS:
        blocking_issues.append("Image is severely overexposed.")
        suggestions.append("Avoid direct flash or strong light and retake the photo.")
    elif brightness > RECOMMENDED_MAX_BRIGHTNESS:
        warnings.append("Image appears somewhat overexposed.")
        suggestions.append("Reduce strong lighting or flash if possible.")

    if contrast < MIN_CONTRAST:
        blocking_issues.append("Image contrast is too low to assess reliably.")
        suggestions.append("Use even lighting and make sure the skin area is visible.")
    elif contrast < RECOMMENDED_CONTRAST:
        warnings.append("Image contrast is somewhat low.")
        suggestions.append("A clearer image with more even lighting may help.")

    acceptable = len(blocking_issues) == 0
    has_warnings = len(warnings) > 0

    if not acceptable:
        message = "Image quality is too poor to submit. Please retake or choose a clearer image."
        status = "needs_retake"
    elif has_warnings:
        message = "Image can be submitted, but its quality may affect the result."
        status = "acceptable_with_warnings"
    else:
        message = "Image quality checks passed. You can submit this image."
        status = "acceptable"

    return {
        "is_acceptable": acceptable,
        "status": status,
        "width": int(width),
        "height": int(height),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "issues": blocking_issues,
        "warnings": warnings,
        "suggestions": suggestions,
        "message": message
    }


# ---------------- REQUEST MODELS ----------------

class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(max_length=150)
    password: str = Field(min_length=8, max_length=128)
    role: str = "user"
    age: int | None = Field(default=None, ge=1, le=120)
    gender: str | None = Field(default=None, max_length=30)
    medical_history: str | None = None
    previous_skin_treatments: str | None = None
    current_medications: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class PatientProfileRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str | None = Field(default=None, max_length=30)
    medical_history: str | None = None
    previous_skin_treatments: str | None = None
    current_medications: str | None = None


# ---------------- BASIC ROUTES ----------------

@app.get("/api")
def api_root():
    return {"message": "DermAI API is running"}


@app.get("/health")
def health():
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")

        return {
            "status": "ok",
            "database": "connected"
        }

    except psycopg.Error:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed."
        )


@app.get("/model/health")
def model_health():
    return {
        "loaded": classifier_model is not None,
        "is_multimodal": is_multimodal,
        "model_name": "EfficientNet-B0 + Clinical Metadata Fusion" if is_multimodal else ("EfficientNet-B0 Balanced" if classifier_model is not None else None),
        "classes": classifier_classes,
        "checkpoint_path": str(FINAL_MODEL_PATH if is_multimodal else STANDALONE_MODEL_PATH)
    }


# ---------------- JWT AUTHENTICATION ----------------

def create_access_token(user_id: int, role: str):

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )

        user_id = int(payload["sub"])

    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token."
        )

    with get_connection() as conn:
        user = conn.execute(
            """
            SELECT user_id, name, email, role
            FROM users
            WHERE user_id = %s
            """,
            (user_id,)
        ).fetchone()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found."
        )

    return {
        "user_id": user[0],
        "name": user[1],
        "email": user[2],
        "role": user[3]
    }


def require_role(*allowed_roles):

    def role_checker(
        user: dict = Depends(get_current_user)
    ):

        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this resource."
            )

        return user

    return role_checker


# ---------------- USER REGISTRATION ----------------

@app.post("/auth/register")
def register(data: RegisterRequest):

    email = data.email.strip().lower()

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Invalid email address.")

    if data.role not in {"user", "clinician"}:
        raise HTTPException(
            status_code=403,
            detail="Only patient and clinician registration is allowed."
        )

    if data.role == "user" and data.age is None:
        raise HTTPException(
            status_code=400,
            detail="Age is required for patient registration."
        )

    hashed_password = password_hash.hash(data.password)
    is_approved = data.role == "user"

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users
                    (name, email, password_hash, role, is_approved)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING user_id, name, email, role, is_approved
                    """,
                    (data.name.strip(), email, hashed_password, data.role, is_approved)
                )
                user = cur.fetchone()

                patient = None
                if data.role == "user":
                    cur.execute(
                        """
                        INSERT INTO patient_profiles
                        (user_id, patient_name, age, gender, medical_history,
                         previous_skin_treatments, current_medications)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING patient_id, user_id, patient_name, age, gender,
                                  medical_history, previous_skin_treatments,
                                  current_medications
                        """,
                        (
                            user[0], data.name.strip(), data.age, data.gender,
                            data.medical_history, data.previous_skin_treatments,
                            data.current_medications
                        )
                    )
                    patient = cur.fetchone()

    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    if data.role == "clinician":
        return {
            "message": "Clinician registration submitted. Your account is waiting for administrator approval.",
            "pending_approval": True,
            "user": {
                "user_id": user[0], "name": user[1], "email": user[2],
                "role": user[3], "is_approved": user[4]
            }
        }

    token = create_access_token(user[0], user[3])
    return {
        "message": "Patient registration successful.",
        "access_token": token,
        "token_type": "bearer",
        "profile_completed": True,
        "user": {
            "user_id": user[0], "name": user[1],
            "email": user[2], "role": user[3]
        },
        "patient": {
            "patient_id": patient[0], "user_id": patient[1],
            "patient_name": patient[2], "age": patient[3],
            "gender": patient[4], "medical_history": patient[5],
            "previous_skin_treatments": patient[6],
            "current_medications": patient[7]
        }
    }


# ---------------- USER LOGIN ----------------

@app.post("/auth/login")
def login(data: LoginRequest):

    with get_connection() as conn:
        user = conn.execute(
            """
            SELECT user_id, name, email, password_hash, role, is_approved
            FROM users
            WHERE email = %s
            """,
            (data.email.strip().lower(),)
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password."
            )

        is_valid = False
        try:
            is_valid = password_hash.verify(data.password, user[3])
        except Exception:
            is_valid = False

        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password."
            )

        if user[4] == "clinician" and not user[5]:
            raise HTTPException(
                status_code=403,
                detail="Your clinician account is waiting for administrator approval."
            )

        profile = (True,) if user[4] != "user" else conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM patient_profiles WHERE user_id = %s
            )
            """,
            (user[0],)
        ).fetchone()

    token = create_access_token(user[0], user[4])

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "profile_completed": profile[0],
        "user": {
            "user_id": user[0],
            "name": user[1],
            "email": user[2],
            "role": user[4]
        }
    }


# ---------------- CURRENT USER PROFILE ----------------

@app.get("/auth/me")
def get_my_profile(
    user: dict = Depends(get_current_user)
):
    return user


# ---------------- PATIENT PROFILE APIs ----------------

@app.post("/patient/profile")
def create_patient_profile(
    data: PatientProfileRequest,
    user: dict = Depends(require_role("user"))
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patient_profiles
                (user_id, patient_name, age, gender, medical_history,
                 previous_skin_treatments, current_medications)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    patient_name = EXCLUDED.patient_name,
                    age = EXCLUDED.age,
                    gender = EXCLUDED.gender,
                    medical_history = EXCLUDED.medical_history,
                    previous_skin_treatments = EXCLUDED.previous_skin_treatments,
                    current_medications = EXCLUDED.current_medications,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING patient_id, user_id, patient_name, age, gender,
                          medical_history, previous_skin_treatments,
                          current_medications, created_at, updated_at
                """,
                (
                    user["user_id"], user["name"], data.age, data.gender,
                    data.medical_history, data.previous_skin_treatments,
                    data.current_medications
                )
            )
            profile = cur.fetchone()

    return {
        "message": "Patient profile saved successfully",
        "profile_completed": True,
        "patient": {
            "patient_id": profile[0], "user_id": profile[1],
            "patient_name": profile[2], "age": profile[3],
            "gender": profile[4], "medical_history": profile[5],
            "previous_skin_treatments": profile[6],
            "current_medications": profile[7],
            "created_at": profile[8], "updated_at": profile[9]
        }
    }


@app.get("/patient/profile")
def get_patient_profile(
    user: dict = Depends(require_role("user"))
):
    with get_connection() as conn:
        profile = conn.execute(
            """
            SELECT patient_id, user_id, patient_name, age, gender,
                   medical_history, previous_skin_treatments,
                   current_medications, created_at, updated_at
            FROM patient_profiles
            WHERE user_id = %s
            """,
            (user["user_id"],)
        ).fetchone()

    if profile is None:
        return {"profile_completed": False, "message": "Patient profile not found."}

    return {
        "profile_completed": True,
        "patient": {
            "patient_id": profile[0], "user_id": profile[1],
            "patient_name": profile[2], "age": profile[3],
            "gender": profile[4], "medical_history": profile[5],
            "previous_skin_treatments": profile[6],
            "current_medications": profile[7],
            "created_at": profile[8], "updated_at": profile[9]
        }
    }


@app.put("/patient/profile")
def update_patient_profile(
    data: PatientProfileRequest,
    user: dict = Depends(require_role("user"))
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE patient_profiles
                SET age = %s, gender = %s, medical_history = %s,
                    previous_skin_treatments = %s,
                    current_medications = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
                RETURNING patient_id, user_id, patient_name, age, gender,
                          medical_history, previous_skin_treatments,
                          current_medications, created_at, updated_at
                """,
                (
                    data.age, data.gender, data.medical_history,
                    data.previous_skin_treatments, data.current_medications,
                    user["user_id"]
                )
            )
            profile = cur.fetchone()

    if profile is None:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    return {
        "message": "Patient profile updated successfully",
        "profile_completed": True,
        "patient": {
            "patient_id": profile[0], "user_id": profile[1],
            "patient_name": profile[2], "age": profile[3],
            "gender": profile[4], "medical_history": profile[5],
            "previous_skin_treatments": profile[6],
            "current_medications": profile[7],
            "created_at": profile[8], "updated_at": profile[9]
        }
    }


# ---------------- CLINICIAN APPROVAL (ADMIN) ----------------

@app.get("/admin/clinicians/pending")
def pending_clinicians(
    admin: dict = Depends(require_role("admin"))
):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, name, email
            FROM users
            WHERE role = 'clinician' AND is_approved = FALSE
            ORDER BY user_id DESC
            """
        ).fetchall()

    return {
        "pending_clinicians": [
            {"user_id": row[0], "name": row[1], "email": row[2]}
            for row in rows
        ]
    }


@app.patch("/admin/clinicians/{clinician_id}/approve")
def approve_clinician(
    clinician_id: int,
    admin: dict = Depends(require_role("admin"))
):
    with get_connection() as conn:
        row = conn.execute(
            """
            UPDATE users
            SET is_approved = TRUE
            WHERE user_id = %s AND role = 'clinician'
            RETURNING user_id, name, email
            """,
            (clinician_id,)
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Clinician account not found.")

    return {
        "message": "Clinician approved successfully.",
        "clinician": {"user_id": row[0], "name": row[1], "email": row[2]}
    }


# ---------------- USER DASHBOARD & CASES ----------------

@app.get("/user/dashboard")
def user_dashboard(user: dict = Depends(require_role("user"))):
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Total analyses
            cur.execute("SELECT count(*) FROM cases WHERE user_id = %s", (user["user_id"],))
            total_cases = cur.fetchone()[0]

            # 2. Previous predictions
            cur.execute("""
                SELECT count(p.prediction_id)
                FROM cases c
                JOIN images i ON c.case_id = i.case_id
                JOIN predictions p ON i.image_id = p.image_id
                WHERE c.user_id = %s
            """, (user["user_id"],))
            total_predictions = cur.fetchone()[0]

            # 3. Recent activity
            cur.execute("""
                SELECT created_at FROM cases
                WHERE user_id = %s
                ORDER BY created_at DESC LIMIT 1
            """, (user["user_id"],))
            recent_act = cur.fetchone()
            recent_activity = recent_act[0].strftime("%b %d, %Y") if recent_act and recent_act[0] else "No recent activity"

            # 4. Recent cases list
            cur.execute("""
                SELECT c.case_id, c.created_at, c.status, c.affected_body_part, c.condition_duration, c.symptoms,
                       c.clinical_notes, c.reviewed_at,
                       p.predicted_class, p.confidence, p.risk_level, p.model_version,
                       i.image_path
                FROM cases c
                LEFT JOIN images i ON c.case_id = i.case_id
                LEFT JOIN predictions p ON i.image_id = p.image_id
                WHERE c.user_id = %s
                ORDER BY c.created_at DESC
                LIMIT 20
            """, (user["user_id"],))

            cases = []
            for r in cur.fetchall():
                img_filename = Path(r[12]).name if r[12] else None
                pred_class = r[8]
                disease_info = DISEASE_METADATA.get(pred_class, {}) if pred_class else {}
                cases.append({
                    "case_id": r[0],
                    "created_at": r[1].strftime("%b %d, %Y, %I:%M %p") if r[1] else "",
                    "status": r[2],
                    "affected_body_part": r[3],
                    "condition_duration": r[4],
                    "symptoms": r[5] if isinstance(r[5], list) else (json.loads(r[5]) if r[5] else []),
                    "clinical_notes": r[6],
                    "reviewed_at": r[7].strftime("%b %d, %Y, %I:%M %p") if r[7] else None,
                    "predicted_class": pred_class,
                    "disease_name": disease_info.get("name", pred_class or "Analyzing"),
                    "confidence": round(float(r[9]) * 100, 2) if r[9] is not None else None,
                    "risk_level": r[10] or disease_info.get("risk_level", "Low Risk"),
                    "model_version": r[11],
                    "image_url": f"/uploads/{img_filename}" if img_filename else None
                })

    return {
        "user": user,
        "total_cases": total_cases,
        "total_predictions": total_predictions,
        "recent_activity": recent_activity,
        "cases": cases
    }


@app.get("/user/cases")
def user_cases(user: dict = Depends(require_role("user"))):
    return user_dashboard(user)


# ---------------- CLINICIAN DASHBOARD & REVIEWS ----------------

class ClinicianReviewRequest(BaseModel):
    status: str = Field(..., description="Review status: reviewed or pending")
    clinical_notes: str = Field("", description="Clinical assessment notes")


@app.get("/clinician/dashboard")
def clinician_dashboard(user: dict = Depends(require_role("clinician"))):
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Total cases
            cur.execute("SELECT count(*) FROM cases")
            total_cases = cur.fetchone()[0]

            # Cases requiring review (pending)
            cur.execute("SELECT count(*) FROM cases WHERE status = 'pending'")
            pending_cases = cur.fetchone()[0]

            # High risk count
            cur.execute("SELECT count(*) FROM predictions WHERE risk_level = 'High Risk'")
            high_risk_cases = cur.fetchone()[0]

            # Disease distribution
            cur.execute("""
                SELECT predicted_class, count(*)
                FROM predictions
                WHERE predicted_class IS NOT NULL
                GROUP BY predicted_class
            """)
            distribution = {row[0]: row[1] for row in cur.fetchall()}

            # All recent cases across platform
            cur.execute("""
                SELECT c.case_id, c.created_at, c.status, c.affected_body_part, c.condition_duration, c.symptoms,
                       c.clinical_notes, c.reviewed_at,
                       p.predicted_class, p.confidence, p.risk_level,
                       u.name as patient_name, u.email as patient_email,
                       pp.age, pp.gender,
                       i.image_path
                FROM cases c
                JOIN users u ON c.user_id = u.user_id
                LEFT JOIN patient_profiles pp ON u.user_id = pp.user_id
                LEFT JOIN images i ON c.case_id = i.case_id
                LEFT JOIN predictions p ON i.image_id = p.image_id
                ORDER BY c.created_at DESC
                LIMIT 50
            """)

            cases = []
            for r in cur.fetchall():
                img_filename = Path(r[15]).name if r[15] else None
                pred_class = r[8]
                disease_info = DISEASE_METADATA.get(pred_class, {}) if pred_class else {}
                cases.append({
                    "case_id": r[0],
                    "created_at": r[1].strftime("%b %d, %Y, %I:%M %p") if r[1] else "",
                    "status": r[2],
                    "affected_body_part": r[3],
                    "condition_duration": r[4],
                    "symptoms": r[5] if isinstance(r[5], list) else (json.loads(r[5]) if r[5] else []),
                    "clinical_notes": r[6],
                    "reviewed_at": r[7].strftime("%b %d, %Y, %I:%M %p") if r[7] else None,
                    "predicted_class": pred_class,
                    "disease_name": disease_info.get("name", pred_class or "Pending"),
                    "confidence": round(float(r[9]) * 100, 2) if r[9] is not None else None,
                    "risk_level": r[10] or disease_info.get("risk_level", "Moderate Risk"),
                    "patient_name": r[11],
                    "patient_email": r[12],
                    "patient_age": r[13],
                    "patient_gender": r[14],
                    "image_url": f"/uploads/{img_filename}" if img_filename else None
                })

    return {
        "user": user,
        "total_cases": total_cases,
        "pending_reviews": pending_cases,
        "high_risk_cases": high_risk_cases,
        "disease_distribution": distribution,
        "cases": cases
    }


@app.get("/clinician/cases")
def clinician_cases(user: dict = Depends(require_role("clinician"))):
    return clinician_dashboard(user)


@app.patch("/clinician/cases/{case_id}/review")
def clinician_review_case(
    case_id: int,
    data: ClinicianReviewRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_role("clinician"))
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE cases
                SET status = %s, clinical_notes = %s, reviewed_by = %s, reviewed_at = CURRENT_TIMESTAMP
                WHERE case_id = %s
                RETURNING case_id, status, clinical_notes, user_id
            """, (data.status, data.clinical_notes.strip(), user["user_id"], case_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Case not found.")

            # Queue automated email alert to the patient regarding clinician review
            try:
                patient_user_id = row[3]
                cur.execute("SELECT name, email FROM users WHERE user_id = %s", (patient_user_id,))
                patient_info = cur.fetchone()
                if patient_info and patient_info[1]:
                    background_tasks.add_task(
                        send_clinician_review_email_alert,
                        patient_info[1],
                        patient_info[0],
                        case_id,
                        user["name"],
                        data.clinical_notes.strip(),
                        data.status
                    )
            except Exception as email_exc:
                print(f"Could not queue clinician review email alert: {email_exc}")

    return {
        "message": f"Case #{case_id} review updated successfully.",
        "case_id": row[0],
        "status": row[1],
        "clinical_notes": row[2]
    }


# ---------------- ADMINISTRATOR DASHBOARD & USER MANAGEMENT ----------------

class UpdateUserRoleRequest(BaseModel):
    role: str = Field(..., description="Role: user, clinician, or admin")


@app.get("/admin/dashboard")
def admin_dashboard(user: dict = Depends(require_role("admin"))):
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Registered users count
            cur.execute("SELECT count(*) FROM users")
            total_users = cur.fetchone()[0]

            # 2. Total analyses
            cur.execute("SELECT count(*) FROM cases")
            total_cases = cur.fetchone()[0]

            # 3. Pending clinician approvals
            cur.execute("SELECT count(*) FROM users WHERE role = 'clinician' AND is_approved = FALSE")
            pending_clinicians = cur.fetchone()[0]

            # 4. Total predictions & distribution
            cur.execute("""
                SELECT predicted_class, count(*)
                FROM predictions
                WHERE predicted_class IS NOT NULL
                GROUP BY predicted_class
            """)
            distribution = {row[0]: row[1] for row in cur.fetchall()}

            # 5. All users list
            cur.execute("""
                SELECT user_id, name, email, role, is_approved, created_at
                FROM users
                ORDER BY created_at DESC
            """)
            users_list = [
                {
                    "user_id": r[0],
                    "name": r[1],
                    "email": r[2],
                    "role": r[3],
                    "is_approved": r[4],
                    "created_at": r[5].strftime("%b %d, %Y") if r[5] else ""
                }
                for r in cur.fetchall()
            ]

    return {
        "user": user,
        "total_users": total_users,
        "total_analyses": total_cases,
        "pending_clinicians_count": pending_clinicians,
        "disease_distribution": distribution,
        "users": users_list,
        "model_info": {
            "name": "EfficientNet-B0 + Clinical Metadata Fusion" if is_multimodal else "EfficientNet-B0 Balanced",
            "classes": classifier_classes,
            "is_multimodal": is_multimodal,
            "train_accuracy": "98.07%",
            "val_accuracy": "77.55%",
            "test_accuracy": "76.74%",
            "test_precision": "73.51%",
            "test_macro_f1": "71.86%",
            "cases_count": "2,298",
            "status": "loaded" if classifier_model is not None else "offline"
        }
    }


@app.get("/admin/users")
def admin_get_users(user: dict = Depends(require_role("admin"))):
    return admin_dashboard(user)


@app.patch("/admin/users/{target_user_id}/role")
def admin_update_user_role(
    target_user_id: int,
    data: UpdateUserRoleRequest,
    user: dict = Depends(require_role("admin"))
):
    if data.role not in {"user", "clinician", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid role specified.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users
                SET role = %s, is_approved = TRUE
                WHERE user_id = %s
                RETURNING user_id, name, email, role
            """, (data.role, target_user_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="User not found.")

    return {
        "message": f"User {row[1]} updated to {data.role}.",
        "user": {"user_id": row[0], "name": row[1], "email": row[2], "role": row[3]}
    }


# ---------------- IMAGE QUALITY CHECK ----------------

@app.post("/assess-image")
async def assess_uploaded_image(
    file: UploadFile = File(...),
    user: dict = Depends(require_role("user"))
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image.")

    contents = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    return {
        "message": "Image quality assessment completed.",
        "quality_assessment": assess_image_quality(contents)
    }


# ---------------- IMAGE UPLOAD WITH CASE DETAILS ----------------

@app.post("/upload-image")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    affected_body_part: str = Form(...),
    condition_duration: str = Form(...),
    symptoms: str = Form(...),
    user: dict = Depends(require_role("user"))
):

    affected_body_part = affected_body_part.strip()
    condition_duration = condition_duration.strip()

    if not affected_body_part or len(affected_body_part) > 100:
        raise HTTPException(
            status_code=400,
            detail="Enter a valid affected body part."
        )

    if not condition_duration or len(condition_duration) > 100:
        raise HTTPException(
            status_code=400,
            detail="Enter how long the skin condition has been present."
        )

    try:
        symptoms_list = json.loads(symptoms)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail='Symptoms must be a JSON list, for example ["itching", "pain"].'
        )

    if not isinstance(symptoms_list, list) or not all(
        isinstance(item, str) and item.strip()
        for item in symptoms_list
    ):
        raise HTTPException(
            status_code=400,
            detail="Symptoms must be a list of non-empty text values."
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The file is empty."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
        quality_assessment = assess_image_quality(contents)
        if not quality_assessment["is_acceptable"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": quality_assessment["message"],
                    "quality_assessment": quality_assessment
                }
            )

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    finally:
        await file.close()

    # Fetch patient profile details for clinical multimodal feature extraction
    clinical_data = {}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT age, gender, medical_history FROM patient_profiles WHERE user_id = %s",
                    (user["user_id"],)
                )
                p_row = cur.fetchone()
                if p_row:
                    if p_row[0] is not None:
                        clinical_data["age"] = float(p_row[0])
                    if p_row[1]:
                        clinical_data["gender"] = str(p_row[1]).upper()
    except Exception as exc:
        print(f"Could not load patient profile for metadata: {exc}")

    # Map symptoms and body region into clinical indicators
    symptoms_lower = [s.lower() for s in symptoms_list]
    if any("itch" in s for s in symptoms_lower):
        clinical_data["itch"] = True
    if any("bleed" in s for s in symptoms_lower):
        clinical_data["bleed"] = True
    if any("grow" in s or "grew" in s for s in symptoms_lower):
        clinical_data["grew"] = True
    if any("hurt" in s or "pain" in s for s in symptoms_lower):
        clinical_data["hurt"] = True
    if any("change" in s for s in symptoms_lower):
        clinical_data["changed"] = True
    if any("elevat" in s or "bump" in s for s in symptoms_lower):
        clinical_data["elevation"] = True
    if affected_body_part:
        clinical_data["region"] = affected_body_part.upper()

    # Classify the quality-approved image using the trained multimodal model.
    prediction = predict_skin_disease(contents, clinical_dict=clinical_data)

    extension = (
        ".png" if file.content_type == "image/png"
        else ".jpg"
    )

    filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                    INSERT INTO cases
                    (user_id, status, affected_body_part,
                     condition_duration, symptoms)
                    VALUES (%s, 'pending', %s, %s, %s)
                    RETURNING case_id
                    """,
                    (
                        user["user_id"],
                        affected_body_part,
                        condition_duration,
                        json.dumps([
                            item.strip() for item in symptoms_list
                        ])
                    )
                )

                case_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO images (case_id, image_path)
                    VALUES (%s, %s)
                    RETURNING image_id
                    """,
                    (case_id, str(destination))
                )

                image_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO predictions (image_id, predicted_class, confidence, model_version, risk_level)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING prediction_id
                    """,
                    (
                        image_id,
                        prediction["predicted_class"],
                        round(prediction["confidence"] / 100.0, 4),
                        prediction.get("model_name", "EfficientNet-B0 Multimodal"),
                        prediction.get("risk_level", "Low Risk")
                    )
                )
                prediction_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO prediction_history (prediction_id)
                    VALUES (%s)
                    """,
                    (prediction_id,)
                )

                destination.write_bytes(contents)

    except (psycopg.Error, OSError) as exc:

        if destination.exists():
            destination.unlink()

        print(f"Upload error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Upload failed. Check the backend terminal."
        ) from exc

    # Dispatch automated email report alert asynchronously in background
    email_dispatched = False
    try:
        if user.get("email"):
            background_tasks.add_task(
                send_analysis_email_alert,
                user["email"],
                user["name"],
                case_id,
                prediction,
                affected_body_part,
                condition_duration
            )
            email_dispatched = True
    except Exception as exc:
        print(f"Failed to queue background email task: {exc}")

    return {
        "message": "Image and case details uploaded successfully",
        "case_id": case_id,
        "image_id": image_id,
        "filename": filename,
        "affected_body_part": affected_body_part,
        "condition_duration": condition_duration,
        "symptoms": symptoms_list,
        "status": "pending",
        "analysis": "completed",
        "prediction": prediction,
        "quality_assessment": quality_assessment,
        "email_notification_sent": email_dispatched,
        "recipient_email": user.get("email")
    }


# ---------------- REPORT EMAIL DISPATCH ENDPOINTS ----------------

@app.post("/cases/{case_id}/send-report-email")
def send_case_report_email(
    case_id: int,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.case_id, c.user_id, c.affected_body_part, c.condition_duration,
                       u.name, u.email,
                       p.predicted_class, p.confidence, p.risk_level
                FROM cases c
                JOIN users u ON c.user_id = u.user_id
                LEFT JOIN images i ON i.case_id = c.case_id
                LEFT JOIN predictions p ON p.image_id = i.image_id
                WHERE c.case_id = %s
                ORDER BY p.prediction_id DESC
                LIMIT 1
            """, (case_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Case not found.")

            # Access control: users can only trigger email for their own cases; clinicians/admins can send for any
            if user["role"] == "user" and row[1] != user["user_id"]:
                raise HTTPException(status_code=403, detail="Unauthorized access to this patient case.")

            pred_class = row[6] or "N/A"
            conf_val = round(float(row[7] * 100), 2) if row[7] is not None else 0.0
            meta = DISEASE_METADATA.get(pred_class, {
                "name": pred_class,
                "category": "Skin Condition",
                "risk_level": row[8] or "Moderate Risk",
                "recommendation": "Consult a physician for clinical assessment."
            })

            pred = {
                "predicted_class": pred_class,
                "disease_name": meta.get("name", pred_class),
                "confidence": conf_val,
                "risk_level": row[8] or meta.get("risk_level", "Moderate Risk"),
                "clinical_recommendation": meta.get("recommendation", "Consult a physician for clinical assessment."),
                "treatment_required": meta.get("treatment_required", "Clinical Assessment Advised")
            }

            recipient_email = row[5]
            patient_name = row[4]
            affected_region = row[2] or "Unspecified"
            duration_str = row[3] or "Unspecified"

            background_tasks.add_task(
                send_analysis_email_alert,
                recipient_email,
                patient_name,
                case_id,
                pred,
                affected_region,
                duration_str
            )

    return {
        "message": f"Report alert successfully dispatched to {recipient_email}.",
        "recipient_email": recipient_email,
        "case_id": case_id
    }


# ---------------- FRONTEND STATIC MOUNT ----------------
FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
