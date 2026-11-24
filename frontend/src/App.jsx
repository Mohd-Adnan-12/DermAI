
import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
  Shield,
  Activity,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Lock,
  UploadCloud,
  Camera,
  FileText,
  Layers,
  Cpu,
  BarChart3,
  Database,
  Sparkles,
  Stethoscope,
  UserCheck,
  Search,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  Info,
  X,
  ExternalLink,
  Menu,
  HeartPulse,
  BrainCircuit,
  Sliders,
  Check,
  User,
  Users,
  Award
} from "lucide-react";

const API_URL = "";

const DISEASE_NAMES_MAP = {
  ACK: "Actinic Keratosis",
  BCC: "Basal Cell Carcinoma",
  MEL: "Melanoma",
  NEV: "Melanocytic Nevus",
  SCC: "Squamous Cell Carcinoma",
  SEK: "Seborrheic Keratosis",
};

const CNN_MODEL_COMPARISON = [
  {
    id: "effnet",
    name: "EfficientNet-B0",
    displayName: "EfficientNet-B0 (Multimodal)",
    releaseYear: 2019,
    params: "~5.3M",
    featureDim: 1280,
    accuracy: 76.74,
    precision: 73.51,
    recall: 70.47,
    macroF1: 71.86,
    trainTime: 918.8,
    infTime: 34.09,
    isProduction: true,
    badge: "Selected Production Model",
    color: "#148074",
    architecture: "Compound scaling with inverted residual MBConv blocks + Squeeze-and-Excitation. Fused with 64 clinical metadata features (320 total).",
    pros: "Highest Macro F1 (71.86%) & Macro Recall (70.47%), fastest inference (34.09 ms), compact footprint, and best sensitivity on malignant minority classes (MEL, SCC)."
  },
  {
    id: "resnet",
    name: "ResNet50",
    displayName: "ResNet50 (Multimodal)",
    releaseYear: 2015,
    params: "~25.6M",
    featureDim: 2048,
    accuracy: 77.83,
    precision: 74.70,
    recall: 64.13,
    macroF1: 67.47,
    trainTime: 936.8,
    infTime: 36.19,
    isProduction: false,
    badge: "Highest Overall Accuracy",
    color: "#e74c3c",
    architecture: "50-layer deep residual network with bottleneck residual building blocks and skip connections. Fused with 64 metadata features (320 total).",
    pros: "Highest overall accuracy (77.83%) and precision (74.70%), but reduced sensitivity on minority malignant classes (Recall: 64.13%)."
  },
  {
    id: "mobilenet",
    name: "MobileNetV2",
    displayName: "MobileNetV2 (Multimodal)",
    releaseYear: 2018,
    params: "~3.4M",
    featureDim: 1280,
    accuracy: 72.39,
    precision: 68.41,
    recall: 67.23,
    macroF1: 67.46,
    trainTime: 953.2,
    infTime: 36.26,
    isProduction: false,
    badge: "Lightweight Edge Architecture",
    color: "#3498db",
    architecture: "Inverted residual structure with linear bottleneck layers and depthwise separable convolutions. Fused with 64 metadata features (320 total).",
    pros: "Smallest parameter footprint (3.4M), suitable for edge mobile deployment; lower test accuracy (72.39%) on subtle dermatological lesions."
  },
  {
    id: "convnext",
    name: "ConvNeXt-Tiny",
    displayName: "ConvNeXt-Tiny (Multimodal)",
    releaseYear: 2022,
    params: "~28.6M",
    featureDim: 768,
    accuracy: 75.65,
    precision: 70.56,
    recall: 66.22,
    macroF1: 67.63,
    trainTime: 941.3,
    infTime: 36.14,
    isProduction: false,
    badge: "Modern ViT-Style CNN",
    color: "#9b59b6",
    architecture: "Modernized pure convolutional network integrating ViT principles: 7x7 depthwise convolutions, inverted bottlenecks, and GELU activations.",
    pros: "Strong global representation learning; competitive 75.65% accuracy and 36.14 ms latency, promising candidate for larger future training corpora."
  }
];

const DATASET_ANALYTICS = {
  name: "PAD-UFES-20 (Dermatological Mobile Images)",
  totalImages: 2298,
  totalPatients: 1373,
  categoriesCount: 6,
  matchedImages: 2298,
  missingMatches: 0,
  duplicateRecords: 0,
  diseaseDistribution: [
    { code: "BCC", name: "Basal Cell Carcinoma", count: 845, pct: 36.77, risk: "High Risk", color: "#e74c3c" },
    { code: "ACK", name: "Actinic Keratosis", count: 730, pct: 31.77, risk: "Moderate Risk", color: "#f39c12" },
    { code: "NEV", name: "Melanocytic Nevus", count: 244, pct: 10.62, risk: "Low Risk", color: "#2ecc71" },
    { code: "SEK", name: "Seborrheic Keratosis", count: 235, pct: 10.23, risk: "Low Risk", color: "#27ae60" },
    { code: "SCC", name: "Squamous Cell Carcinoma", count: 192, pct: 8.35, risk: "High Risk", color: "#c0392b" },
    { code: "MEL", name: "Melanoma", count: 52, pct: 2.26, risk: "High Risk", color: "#8e44ad" },
  ],
  splits: [
    { name: "Training Partition", count: 1397, pct: 60.8, color: "#148074", note: "Patient-wise grouped with class-weighted loss" },
    { name: "Validation Partition", count: 441, pct: 19.2, color: "#3498db", note: "Unbiased checkpointing and lr scheduler tracking" },
    { name: "Testing Partition", count: 460, pct: 20.0, color: "#f39c12", note: "Completely held-out evaluation test set" }
  ]
};


function getReadableError(detail, fallback) {
  if (typeof detail === "string" && detail.trim()) return detail;

  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        return item.msg || item.message || JSON.stringify(item);
      }
      return String(item);
    });
    return messages.filter(Boolean).join("; ") || fallback;
  }

  if (detail && typeof detail === "object") {
    return detail.message || detail.msg || JSON.stringify(detail);
  }

  return fallback;
}


const DISEASE_DETAILS = {
  BCC: {
    code: "BCC",
    name: "Basal Cell Carcinoma",
    risk: "High Risk",
    riskColor: "#ef4444",
    type: "Malignant Skin Cancer",
    datasetCount: "845 cases (36.8%)",
    description: "The most common non-melanoma skin cancer globally. Arises from basal cells of the lower epidermis. Characterized by slow, localized expansion and pearly papules with prominent telangiectasias.",
    keyFeatures: [
      "Pearly or translucent rolled borders with central depression",
      "Arborizing telangiectatic blood vessels visible under illumination",
      "May ulcerate, bleed, or develop crusting",
      "Very low metastatic potential but causes significant local tissue damage if untreated"
    ],
    triageAction: "Prompt dermatological consultation for biopsy and surgical excision or Mohs micrographic surgery."
  },
  MEL: {
    code: "MEL",
    name: "Melanoma",
    risk: "Critical Urgent Risk",
    riskColor: "#dc2626",
    type: "Malignant Melanocyte Neoplasm",
    datasetCount: "52 cases (2.3%) | 97.69% ROC-AUC",
    description: "The most aggressive cutaneous malignancy, arising from transformed pigment-producing melanocytes. Early diagnosis and prompt excision before deep vertical invasion are vital for survival.",
    keyFeatures: [
      "Asymmetry: Two halves of the lesion do not match",
      "Border: Scalloped, notched, or poorly demarcated borders",
      "Color: Variegated shades of brown, black, blue, pink, or white",
      "Diameter & Evolution: Greater than 6mm and changing over time"
    ],
    triageAction: "Immediate expedited clinical referral (within 1-2 weeks) and complete diagnostic excisional biopsy."
  },
  SCC: {
    code: "SCC",
    name: "Squamous Cell Carcinoma",
    risk: "High Risk",
    riskColor: "#ea580c",
    type: "Malignant Epidermal Carcinoma",
    datasetCount: "192 cases (8.4%)",
    description: "Second most prevalent skin malignancy, originating from keratinocytes of the epidermis. Often emerges from pre-existing actinic keratoses on chronic sun-damaged skin.",
    keyFeatures: [
      "Thick, rough, hyperkeratotic or scaly red plaque",
      "Crusted central ulceration with elevated indurated borders",
      "Frequently tender, painful, or bleeds upon minor trauma",
      "Moderate potential for metastasis to regional lymph nodes"
    ],
    triageAction: "Urgent dermatological assessment for complete surgical resection with negative histological margins."
  },
  NEV: {
    code: "NEV",
    name: "Melanocytic Nevus",
    risk: "Low Risk",
    riskColor: "#10b981",
    type: "Benign Melanocytic Proliferation",
    datasetCount: "244 cases (10.6%)",
    description: "Common benign proliferation of melanocytes (moles). Characterized by regular symmetrical architecture, uniform pigmentation, and smooth borders.",
    keyFeatures: [
      "Symmetric circular or oval geometry",
      "Uniform light-to-dark brown pigmentation",
      "Sharp, well-demarcated peripheral margins",
      "Stable over time without rapid morphologic alterations"
    ],
    triageAction: "Routine self-monitoring and periodic baseline clinical skin examinations."
  },
  ACK: {
    code: "ACK",
    name: "Actinic Keratosis",
    risk: "Moderate Risk",
    riskColor: "#f59e0b",
    type: "Pre-Cancerous Keratinocytic Dysplasia",
    datasetCount: "730 cases (31.8%)",
    description: "Pre-malignant, sun-induced epidermal lesion exhibiting atypical keratinocyte proliferation. Recognized as a precursor lesion that can progress to invasive Squamous Cell Carcinoma if untreated.",
    keyFeatures: [
      "Dry, rough, sandpaper-like scaly texture",
      "Erythematous base on chronically sun-exposed areas (face, scalp, forearms)",
      "Prickling or burning sensation reported by patients",
      "Estimated 0.1% to 10% annual transformation rate to invasive SCC"
    ],
    triageAction: "Cryotherapy, topical 5-fluorouracil, imiquimod, or photodynamic therapy recommended."
  },
  SEK: {
    code: "SEK",
    name: "Seborrheic Keratosis",
    risk: "Low Risk",
    riskColor: "#059669",
    type: "Benign Epithelial Tumor",
    datasetCount: "235 cases (10.2%)",
    description: "Extremely common benign non-cancerous epithelial skin growth appearing with increasing age. Completely non-malignant with zero transformation risk.",
    keyFeatures: [
      "Distinct 'stuck-on' waxy or verrucous appearance",
      "Color ranges from flesh-toned to tan, brown, or black",
      "Keratin pseudocysts or follicular plugs visible on surface",
      "No invasive potential; cosmetic or irritation concern only"
    ],
    triageAction: "Benign clinical reassurance. Cryotherapy or curettage if irritated by clothing or friction."
  }
};

const WORKFLOW_STEPS = [
  {
    step: "01",
    title: "Image Acquisition",
    desc: "Capture a close-up photo using a smartphone camera or upload an existing image with standard lighting.",
    icon: Camera
  },
  {
    step: "02",
    title: "Quality Verification",
    desc: "OpenCV algorithms automatically evaluate sharpness (Laplacian variance), illumination, and contrast before analysis.",
    icon: CheckCircle2
  },
  {
    step: "03",
    title: "Clinical Metadata Fusion",
    desc: "Incorporate patient age, anatomical location, lesion itch, bleeding, and elevation to contextualize the analysis.",
    icon: Sliders
  },
  {
    step: "04",
    title: "Multimodal Deep Learning",
    desc: "EfficientNet-B0 extracts 1,280 visual features, fused with 64 metadata dimensions for calibrated probability scoring.",
    icon: Cpu
  },
  {
    step: "05",
    title: "Grad-CAM Heatmaps",
    desc: "Gradient-weighted Class Activation Mapping highlights the exact lesion regions that influenced the model's decision.",
    icon: Eye
  },
  {
    step: "06",
    title: "Triage & Clinical Report",
    desc: "The system assigns a risk level (Low, Moderate, High, Urgent), suggests clinical guidelines, and exports a signed report.",
    icon: FileText
  }
];

const TECH_STACK = [
  { name: "React 19 & Vite", role: "Frontend Engine", desc: "Ultra-fast client interface with responsive component architecture", icon: Activity },
  { name: "FastAPI & Python 3.11", role: "Backend API", desc: "High-performance asynchronous REST endpoints with Pydantic validation", icon: Cpu },
  { name: "PyTorch & EfficientNet-B0", role: "Deep Learning", desc: "Compound-scaled multimodal CNN pretrained on ImageNet and fine-tuned on PAD-UFES-20", icon: BrainCircuit },
  { name: "OpenCV & PIL", role: "Computer Vision", desc: "Automated image sharpness, illumination, contrast gating, and preprocessing", icon: Eye },
  { name: "Grad-CAM", role: "Explainable AI", desc: "Layer-wise activation gradient visualization for diagnostic transparency", icon: Sparkles },
  { name: "PostgreSQL & Psycopg 3", role: "Database", desc: "Relational persistence with secure password hashing and role-based permissions", icon: Database },
  { name: "Docker & Containerization", role: "Deployment", desc: "Isolated, reproducible multi-stage Docker build deployed on Render cloud", icon: Layers },
  { name: "Brevo HTTPS API", role: "Email Notifications", desc: "Port 443 encrypted transactional email delivery for clinical alerts and password resets", icon: Shield }
];

const MODEL_METRICS = [
  { label: "Training Accuracy", value: "98.07%", note: "Patient-grouped partitioned training", color: "#0d9488" },
  { label: "Validation Accuracy", value: "77.55%", note: "Unbiased checkpoint evaluation", color: "#0284c7" },
  { label: "Test Accuracy", value: "76.74%", note: "Held-out test set evaluation", color: "#6366f1" },
  { label: "Macro Precision", value: "73.51%", note: "Across all 6 disease classes", color: "#8b5cf6" },
  { label: "Macro Recall", value: "70.47%", note: "Superior sensitivity on malignancies", color: "#ec4899" },
  { label: "Macro F1-Score", value: "71.86%", note: "Harmonic mean of precision & recall", color: "#10b981" },
  { label: "Macro ROC-AUC", value: "90.31%", note: "Multi-class discriminative capacity", color: "#f59e0b" },
  { label: "Melanoma ROC-AUC", value: "97.69%", note: "Exceptional malignant detection", color: "#ef4444" }
];

function App() {
  const [page, setPage] = useState("landing");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [resetToken, setResetToken] = useState("");
  const [resetNewPassword, setResetNewPassword] = useState("");
  const [resetConfirmPassword, setResetConfirmPassword] = useState("");
  const [showResetPassword, setShowResetPassword] = useState(false);
  const [activeDiseaseModal, setActiveDiseaseModal] = useState(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const tokenFromUrl = urlParams.get("reset_token");
    if (tokenFromUrl) {
      setResetToken(tokenFromUrl);
      setPage("reset_password");
    }
  }, []);

  const [user, setUser] = useState(null);
  const [token, setToken] = useState("");
  const [activePage, setActivePage] = useState("Dashboard");

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("user");

  // Patient medical profile
  const [patientProfile, setPatientProfile] = useState(null);
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientGender, setPatientGender] = useState("");
  const [medicalHistory, setMedicalHistory] = useState("");
  const [previousSkinTreatments, setPreviousSkinTreatments] = useState("");
  const [currentMedications, setCurrentMedications] = useState("");
  const [patientEditing, setPatientEditing] = useState(false);
  const [patientReturnToProfile, setPatientReturnToProfile] = useState(false);

  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingClinicians, setPendingClinicians] = useState([]);

  // Dashboard data for all roles
  const [dashboardData, setDashboardData] = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);

  // Clinician review state
  const [selectedCase, setSelectedCase] = useState(null);
  const [reviewNotes, setReviewNotes] = useState("");
  const [reviewStatus, setReviewStatus] = useState("reviewed");
  const [activeChartTab, setActiveChartTab] = useState("accuracy");
  const [mlAnalyticsSubSection, setMlAnalyticsSubSection] = useState("all");
  const [clinicianCaseFilter, setClinicianCaseFilter] = useState("all");

  const [selectedImage, setSelectedImage] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [qualityAssessment, setQualityAssessment] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [lastAnalyzedImage, setLastAnalyzedImage] = useState(null);
  const [isAssessingQuality, setIsAssessingQuality] = useState(false);
  const [emailAlertInfo, setEmailAlertInfo] = useState(null);
  const [isSendingEmail, setIsSendingEmail] = useState(false);

  // Skin analysis details (collected separately for every case)
  const [affectedBodyPart, setAffectedBodyPart] = useState("");
  const [conditionDuration, setConditionDuration] = useState("");
  const [symptoms, setSymptoms] = useState([]);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);

  useEffect(() => () => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  // Auto-load dashboard data when user logs in or navigates to a data page
  const DATA_PAGES = [
    "Dashboard", "Previous Predictions", "Prediction History",
    "Image Results", "Reports", "Notifications & Alerts",
    "Total Cases", "Disease Distribution",
    "Prediction Confidence", "Model Performance",
    "ML Analytics", "Registered Users", "Total Analyses", "Platform Health",
  ];
  useEffect(() => {
    if (token && user && DATA_PAGES.includes(activePage)) {
      loadDashboard();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, user, activePage]);

  // Section 17: Notification & Alert System State
  const [notifications, setNotifications] = useState([
    {
      id: "notif-1",
      title: "AI Analysis Completed",
      message: "Case #1042 has completed multimodal classification with 88.4% confidence.",
      level: "Informational",
      timestamp: "10 mins ago",
      type: "analysis_completed",
      read: false,
    },
    {
      id: "notif-2",
      title: "Clinical Review Recommended",
      message: "High-risk lesion (Melanoma suspicion) detected. In-person biopsy advised.",
      level: "High",
      timestamp: "1 hour ago",
      type: "clinical_review",
      read: false,
    },
    {
      id: "notif-3",
      title: "Clinical Report Generated",
      message: "Diagnostic summary report is ready for download in PDF, CSV, and JSON.",
      level: "Informational",
      timestamp: "2 hours ago",
      type: "report_generated",
      read: true,
    },
    {
      id: "notif-4",
      title: "Follow-Up Reminder",
      message: "Routine 30-day lesion monitoring recommended for Case #1039.",
      level: "Low",
      timestamp: "1 day ago",
      type: "follow_up",
      read: true,
    },
  ]);
  const [notifTrayOpen, setNotifTrayOpen] = useState(false);
  const [notifFilter, setNotifFilter] = useState("all");

  function addNotification({ title, message, level = "Informational", type = "general" }) {
    const newNotif = {
      id: `notif-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
      title,
      message,
      level,
      timestamp: "Just now",
      type,
      read: false,
    };
    setNotifications((prev) => [newNotif, ...prev]);
  }

  function markAllNotificationsRead() {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }

  function clearNotification(id) {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }

  // Section 15: Automated Report Downloads (PDF, CSV, JSON)
  function downloadBlob(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function exportCaseJSON(c) {
    if (!c) return;
    const exportData = {
      report_title: "DermAI Dermatological Screening Report",
      export_version: "2.4-multimodal",
      generated_at: new Date().toISOString(),
      case_id: c.case_id || "Active-Prediction",
      patient: {
        name: user?.name || "Patient",
        email: user?.email || "",
        affected_body_part: c.affected_body_part || affectedBodyPart || "Unspecified",
        condition_duration: c.condition_duration || conditionDuration || "Unspecified",
        symptoms: c.symptoms || symptoms || [],
      },
      diagnostic_analysis: {
        predicted_condition: c.disease_name || DISEASE_NAMES_MAP[c.predicted_class] || c.predicted_class,
        disease_code: c.predicted_class,
        disease_category: c.disease_category || "",
        confidence_percentage: c.confidence,
        risk_level: c.risk_level || "Clinical Review Recommended",
        condition_severity: c.severity || c.risk_level,
        treatment_required: c.treatment_required || "Evaluation required",
        recommended_procedure: c.treatment_type || c.clinical_recommendation,
        class_probabilities: c.class_probabilities || {},
      },
      multimodal_model: {
        architecture: "EfficientNet-B0 + 64-dim Clinical Metadata Fusion",
        benchmark_accuracy: "76.74%",
        macro_f1: "71.86%",
        inference_latency: "34.09 ms",
      },
      medical_disclaimer:
        "DermAI provides preliminary AI-assisted clinical decision-support only. This output does not replace professional dermatological examination.",
    };
    const filename = `DermAI_Report_${c.case_id ? `Case_${c.case_id}` : "Prediction"}_${new Date().toISOString().slice(0, 10)}.json`;
    downloadBlob(JSON.stringify(exportData, null, 2), filename, "application/json");
  }

  function exportCaseCSV(c) {
    if (!c) return;
    const headers = [
      "Case_ID",
      "Date",
      "Patient_Name",
      "Patient_Email",
      "Body_Part",
      "Duration",
      "Disease_Code",
      "Disease_Name",
      "Confidence_Pct",
      "Risk_Level",
      "Treatment_Required",
      "Recommended_Treatment",
      "Status"
    ];
    const row = [
      c.case_id || "NEW",
      `"${c.created_at || new Date().toLocaleDateString()}"`,
      `"${user?.name || ""}"`,
      `"${user?.email || ""}"`,
      `"${c.affected_body_part || affectedBodyPart || ""}"`,
      `"${c.condition_duration || conditionDuration || ""}"`,
      `"${c.predicted_class || ""}"`,
      `"${c.disease_name || DISEASE_NAMES_MAP[c.predicted_class] || ""}"`,
      c.confidence || "",
      `"${c.risk_level || ""}"`,
      `"${c.treatment_required || ""}"`,
      `"${(c.treatment_type || c.clinical_recommendation || "").replace(/"/g, '""')}"`,
      `"${c.status || "completed"}"`
    ];
    const csvContent = "\uFEFF" + [headers.join(","), row.join(",")].join("\n");
    const filename = `DermAI_Report_${c.case_id ? `Case_${c.case_id}` : "Prediction"}.csv`;
    downloadBlob(csvContent, filename, "text/csv;charset=utf-8;");
  }

  function exportAllCasesCSV(cases) {
    if (!cases || cases.length === 0) {
      alert("No cases available to export.");
      return;
    }
    const headers = [
      "Case_ID",
      "Date",
      "Patient_Name",
      "Body_Part",
      "Duration",
      "Disease_Code",
      "Disease_Name",
      "Confidence_Pct",
      "Risk_Level",
      "Status"
    ];
    const rows = cases.map((c) =>
      [
        c.case_id,
        `"${c.created_at || ""}"`,
        `"${user?.name || ""}"`,
        `"${c.affected_body_part || ""}"`,
        `"${c.condition_duration || ""}"`,
        `"${c.predicted_class || ""}"`,
        `"${c.disease_name || DISEASE_NAMES_MAP[c.predicted_class] || ""}"`,
        c.confidence || "",
        `"${c.risk_level || ""}"`,
        `"${c.status || "pending"}"`
      ].join(",")
    );
    const csvContent = "\uFEFF" + [headers.join(","), ...rows].join("\n");
    downloadBlob(csvContent, `DermAI_All_Cases_${new Date().toISOString().slice(0, 10)}.csv`, "text/csv;charset=utf-8;");
  }

  async function handleResendEmailReport(caseId) {
    const targetCaseId = caseId || emailAlertInfo?.caseId || analysisResult?.case_id;
    if (!targetCaseId) {
      alert("No case ID available to send report email.");
      return;
    }
    setIsSendingEmail(true);
    try {
      const response = await fetch(`${API_URL}/cases/${targetCaseId}/send-report-email`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getReadableError(data.detail, "Failed to send email alert."));
      }
      const recipient = data.recipient_email || user?.email;
      setEmailAlertInfo({
        caseId: targetCaseId,
        recipientEmail: recipient,
        timestamp: new Date().toLocaleTimeString(),
      });
      addNotification({
        title: "✉️ Email Report Alert Dispatched",
        message: `Diagnostic summary for Case #${targetCaseId} was dispatched directly to ${recipient}.`,
        level: "Informational",
        type: "email_sent",
      });
      alert(`Diagnostic report alert has been emailed to ${recipient}!`);
    } catch (err) {
      alert(`Email alert error: ${err.message}`);
    } finally {
      setIsSendingEmail(false);
    }
  }

  function exportCasePDF(c, gradcamImg) {
    if (!c) return;
    const win = window.open("", "_blank");
    if (!win) return;
    const diseaseName = c.disease_name || DISEASE_NAMES_MAP[c.predicted_class] || c.predicted_class || "Unclassified";
    const risk = c.risk_level || "Clinical Review";
    const conf = c.confidence || 0;
    const caseId = c.case_id ? `#${c.case_id}` : "Live Screening";
    const bodyPart = c.affected_body_part || affectedBodyPart || "Not specified";
    const duration = c.condition_duration || conditionDuration || "Not specified";
    const reportedSyms = Array.isArray(c.symptoms) ? c.symptoms.join(", ") : (symptoms || []).join(", ") || "None";
    const treatment = c.treatment_type || c.clinical_recommendation || "Consult dermatologist for evaluation";
    const imgUrl = gradcamImg || c.gradcam_image || c.image_url || lastAnalyzedImage || "";

    win.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8"/>
        <title>DermAI Clinical Report — ${caseId}</title>
        <style>
          @page { size: A4; margin: 20mm; }
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; padding: 24px; color: #1e293b; max-width: 820px; margin: 0 auto; line-height: 1.5; }
          .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #148074; padding-bottom: 14px; margin-bottom: 20px; }
          .brand-title { color: #0b554d; font-size: 26px; font-weight: 800; margin: 0; }
          .subbrand { color: #64748b; font-size: 13px; margin: 3px 0 0; }
          .doc-meta { text-align: right; font-size: 12px; color: #64748b; }
          .badge-risk { display: inline-block; padding: 5px 14px; border-radius: 14px; font-weight: 700; font-size: 13px;
            background: ${risk.toLowerCase().includes("high") ? "#fee2e2" : risk.toLowerCase().includes("moderate") ? "#fef3c7" : "#dcfce7"};
            color: ${risk.toLowerCase().includes("high") ? "#991b1b" : risk.toLowerCase().includes("moderate") ? "#92400e" : "#166534"}; }
          h2 { color: #0b554d; font-size: 16px; margin: 24px 0 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
          table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
          td { padding: 8px 12px; border: 1px solid #e2e8f0; }
          td:first-child { font-weight: 600; background: #f8fafc; width: 32%; color: #334155; }
          .conf-bar { background: #e2e8f0; border-radius: 6px; height: 12px; overflow: hidden; margin-top: 6px; }
          .conf-fill { background: ${parseFloat(conf) >= 80 ? "#10b981" : parseFloat(conf) >= 60 ? "#f59e0b" : "#ef4444"}; height: 100%; width: ${conf}%; }
          .img-preview { max-width: 220px; max-height: 220px; border-radius: 8px; border: 1px solid #cbd5e1; margin-top: 8px; }
          .treatment-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; padding: 12px 16px; border-radius: 6px; margin-top: 10px; }
          .disclaimer-box { margin-top: 36px; padding: 12px 16px; background: #f8fafc; border: 1px dashed #94a3b8; border-radius: 6px; font-size: 11px; color: #64748b; line-height: 1.6; }
          .print-btn-bar { margin: 20px 0; text-align: right; }
          .btn-print { background: #148074; color: white; border: none; padding: 10px 22px; border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 14px; }
          @media print { .print-btn-bar { display: none; } }
        </style>
      </head>
      <body>
        <div class="print-btn-bar">
          <button class="btn-print" onclick="window.print()">🖨️ Save as PDF / Print</button>
        </div>
        <div class="header">
          <div>
            <h1 class="brand-title">DermAI Clinical Screening Report</h1>
            <p class="subbrand">Multimodal Deep Learning Dermatological Decision Support</p>
          </div>
          <div class="doc-meta">
            <strong>Case Reference:</strong> ${caseId}<br/>
            <strong>Date Generated:</strong> ${new Date().toLocaleDateString()}<br/>
            <strong>Model:</strong> EfficientNet-B0 Multimodal (76.74% Acc)
          </div>
        </div>

        <h2>1. Patient & Case Profile</h2>
        <table>
          <tr><td>Patient Name</td><td>${user?.name || "Patient"}</td></tr>
          <tr><td>Patient Email</td><td>${user?.email || "—"}</td></tr>
          <tr><td>Affected Lesion Location</td><td>${bodyPart}</td></tr>
          <tr><td>Reported Duration</td><td>${duration}</td></tr>
          <tr><td>Observed Symptoms</td><td>${reportedSyms}</td></tr>
        </table>

        <h2>2. Diagnostic Prediction & Classification</h2>
        <table>
          <tr><td>Primary Prediction</td><td><strong>${diseaseName}</strong> (${c.predicted_class || "—"})</td></tr>
          <tr><td>Model Confidence</td><td><strong>${conf}%</strong><div class="conf-bar"><div class="conf-fill"></div></div></td></tr>
          <tr><td>Clinical Risk Level</td><td><span class="badge-risk">${risk}</span></td></tr>
          <tr><td>Condition Severity</td><td>${c.severity || risk}</td></tr>
          <tr><td>Treatment Requirement</td><td>${c.treatment_required || "Clinical assessment recommended"}</td></tr>
        </table>

        <h2>3. Recommended Clinical Action</h2>
        <div class="treatment-box">
          <strong>Next Steps:</strong>
          <p style="margin: 4px 0 0 0;">${treatment}</p>
        </div>

        ${imgUrl ? `
          <h2>4. Lesion Inspection & Explainable AI (Grad-CAM)</h2>
          <img src="${imgUrl}" class="img-preview" alt="Lesion / Grad-CAM visual" />
        ` : ""}

        <div class="disclaimer-box">
          <strong>Regulatory & Clinical Disclaimer:</strong> This document is generated by an artificial intelligence screening system (DermAI Multimodal Classifier). It is designed solely as a clinical decision-support and educational tool, not an autonomous medical diagnosis. Please seek formal clinical evaluation by a certified dermatologist.
        </div>
      </body>
      </html>
    `);
    win.document.close();
  }

  async function startCamera() {
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access is unavailable. Please use Choose from Device.");
      setCameraOpen(true);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } }, audio: false
      });
      cameraStreamRef.current = stream;
      setCameraOpen(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() =>
            setCameraError("Unable to start camera preview. Check browser permissions.")
          );
        }
      });
    } catch (error) {
      setCameraError(error.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in Safari settings."
        : "Could not open the camera. Check that it is connected and not in use.");
      setCameraOpen(true);
    }
  }

  function stopCamera() {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOpen(false);
    setCameraError("");
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video?.videoWidth || !video?.videoHeight) {
      setCameraError("Camera is not ready yet. Please wait and try again.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setCameraError("Unable to capture the photo. Please try again.");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("Unable to create the photo. Please try again.");
        return;
      }
      const file = new File([blob], `dermai-capture-${Date.now()}.jpg`, { type: "image/jpeg" });
      if (selectedImage?.preview) URL.revokeObjectURL(selectedImage.preview);
      setSelectedImage({ file, preview: URL.createObjectURL(file) });
      setUploadStatus("");
      setAnalysisResult(null);
      assessSelectedImage(file);
      stopCamera();
    }, "image/jpeg", 0.92);
  }

  // ---------------- AUTHENTICATION ----------------

  async function handleLogin(event, portalRoleOverride) {
    if (event && event.preventDefault) event.preventDefault();
    setLoading(true);
    setMessage("");

    const effectiveRole = portalRoleOverride || (
      page === "login_patient" ? "patient" :
      page === "login_clinician" ? "clinician" :
      page === "login_admin" ? "admin" : null
    );

    try {
      const payload = { email: email.trim(), password };
      if (effectiveRole) {
        payload.portal_role = effectiveRole;
      }

      const response = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Login failed.");
      }

      setToken(data.access_token);
      setUser(data.user);
      setActivePage("Dashboard");
      setPassword("");

      if (data.user.role === "user") {
        const profileResponse = await fetch(
          `${API_URL}/patient/profile`,
          {
            headers: {
              Authorization: `Bearer ${data.access_token}`,
            },
          }
        );

        const profileData = await profileResponse.json();

        if (!profileResponse.ok) {
          throw new Error(
            profileData.detail || "Could not load patient profile."
          );
        }

        setPatientProfile(profileData.patient || null);

        if (profileData.patient) {
          setPatientName(profileData.patient.patient_name || data.user.name || "");
          setPatientAge(String(profileData.patient.age || ""));
          setPatientGender(profileData.patient.gender || "");
          setMedicalHistory(profileData.patient.medical_history || "");
          setPreviousSkinTreatments(profileData.patient.previous_skin_treatments || "");
          setCurrentMedications(profileData.patient.current_medications || "");
        } else {
          setPatientName(data.user.name || "");
          setPatientAge("");
          setPatientGender("");
          setMedicalHistory("");
          setPreviousSkinTreatments("");
          setCurrentMedications("");
        }

        setPatientEditing(false);
        setPatientReturnToProfile(false);

        setPage(
          profileData.profile_completed ? "dashboard" : "patient"
        );
      } else {
        setPatientProfile(null);
        setPage("dashboard");
      }
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  // ---------------- REGISTRATION ----------------

  async function handleRegister(event) {
    event.preventDefault();
    setLoading(true);
    setMessage("");

    if (password !== confirmPassword) {
      setMessage("Passwords do not match.");
      setLoading(false);
      return;
    }

    const payload = {
      name: name.trim(),
      email: email.trim(),
      password,
      role,
    };

    if (role === "user") {
      payload.age = Number(patientAge);
      payload.gender = patientGender || null;
      payload.medical_history = medicalHistory.trim() || null;
      payload.previous_skin_treatments = previousSkinTreatments.trim() || null;
      payload.current_medications = currentMedications.trim() || null;
    }

    try {
      const response = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Registration failed.");
      }

      setPassword("");
      setConfirmPassword("");
      setName("");

      if (role === "clinician") {
        setMessage(data.message || "Registration submitted for administrator approval.");
        setPage("login");
        return;
      }

      setToken(data.access_token);
      setUser(data.user);
      setPatientProfile(data.patient || null);
      setPatientName(data.patient?.patient_name || data.user.name || "");
      setPatientAge(String(data.patient?.age || ""));
      setPatientGender(data.patient?.gender || "");
      setMedicalHistory(data.patient?.medical_history || "");
      setPreviousSkinTreatments(data.patient?.previous_skin_treatments || "");
      setCurrentMedications(data.patient?.current_medications || "");
      setActivePage("Dashboard");
      setPage("dashboard");
      setEmail(data.user.email);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  // ---------------- FORGOT & RESET PASSWORD ----------------

  async function handleForgotPassword(event) {
    if (event && event.preventDefault) event.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const response = await fetch(`${API_URL}/auth/forgot-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim() }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Password reset request failed.");
      }
      setMessage(data.message || "If an account matches this email, reset instructions have been sent.");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleResetPassword(event) {
    if (event && event.preventDefault) event.preventDefault();
    setLoading(true);
    setMessage("");

    if (!resetToken.trim()) {
      setMessage("Reset token is required.");
      setLoading(false);
      return;
    }

    if (resetNewPassword.length < 6) {
      setMessage("New password must be at least 6 characters.");
      setLoading(false);
      return;
    }

    if (resetNewPassword !== resetConfirmPassword) {
      setMessage("Passwords do not match.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: resetToken.trim(),
          new_password: resetNewPassword,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Password reset failed.");
      }

      setMessage("Password successfully reset! Please sign in with your new credentials.");
      setResetNewPassword("");
      setResetConfirmPassword("");
      setResetToken("");
      setPage("login_patient");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ---------------- SIGN OUT ----------------

  function handleSignOut() {
    setToken("");
    setUser(null);
    setPage("landing");
    setActivePage("Dashboard");
    setPassword("");
    setConfirmPassword("");
    setMessage("");
    setSelectedImage(null);
    setUploadStatus("");
    setAnalysisResult(null);
    setAffectedBodyPart("");
    setConditionDuration("");
    setSymptoms([]);
    setPatientProfile(null);
    setPatientName("");
    setPatientAge("");
    setPatientGender("");
    setMedicalHistory("");
    setPreviousSkinTreatments("");
    setCurrentMedications("");
    setPatientEditing(false);
    setPatientReturnToProfile(false);
  }

  // ---------------- PATIENT PROFILE ----------------

  async function handlePatientSubmit(event) {
    event.preventDefault();

    if (!token) {
      setMessage("Your session has expired. Please sign in again.");
      setPage("login");
      return;
    }

    setLoading(true);
    setMessage("");

    const payload = {
      age: Number(patientAge),
      gender: patientGender || null,
      medical_history: medicalHistory.trim() || null,
      previous_skin_treatments: previousSkinTreatments.trim() || null,
      current_medications: currentMedications.trim() || null,
    };

    try {
      const method = patientProfile ? "PUT" : "POST";

      const response = await fetch(`${API_URL}/patient/profile`, {
        method,
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not save patient details."
        );
      }

      setPatientProfile(data.patient);
      setPatientEditing(false);
      setMessage("");

      if (patientReturnToProfile) {
        setActivePage("Profile");
        setPage("dashboard");
      } else {
        setActivePage("Dashboard");
        setPage("dashboard");
      }

      setPatientReturnToProfile(false);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  function editPatientProfile() {
    setPatientName(user?.name || "");

    setPatientAge(
      patientProfile?.age
        ? String(patientProfile.age)
        : ""
    );

    setPatientGender(patientProfile?.gender || "");
    setMedicalHistory(patientProfile?.medical_history || "");
    setPreviousSkinTreatments(patientProfile?.previous_skin_treatments || "");
    setCurrentMedications(patientProfile?.current_medications || "");

    setPatientEditing(true);
    setPatientReturnToProfile(true);
    setMessage("");
    setPage("patient");
  }

  // ---------------- IMAGE QUALITY ASSESSMENT ----------------

  async function assessSelectedImage(file) {
    if (!file) return;

    setQualityAssessment(null);
    setIsAssessingQuality(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/assess-image`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(getReadableError(data.detail, "Image quality assessment failed."));
      }

      const qa = data.quality_assessment || null;
      setQualityAssessment(qa);

      if (qa && !qa.is_acceptable) {
        addNotification({
          title: "Poor Image Quality Warning",
          message: "The uploaded skin image has low contrast or blur. Please ensure bright illumination and sharp focus.",
          level: "Medium",
          type: "poor_quality",
        });
      }
    } catch (error) {
      setQualityAssessment({
        is_acceptable: false,
        status: "assessment_error",
        message: error.message || "Could not assess image quality. Please try again.",
        issues: [error.message || "Could not assess image quality."],
        suggestions: [],
      });
      addNotification({
        title: "Quality Assessment Issue",
        message: "Automated quality check failed. Please verify image format and resolution.",
        level: "Low",
        type: "poor_quality",
      });
    } finally {
      setIsAssessingQuality(false);
    }
  }

  // ---------------- IMAGE UPLOAD ----------------

  function handleImageSelect(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!["image/jpeg", "image/png"].includes(file.type)) {
      setUploadStatus("Please select a JPG or PNG image.");
      event.target.value = "";
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadStatus("Image size must be 10 MB or smaller.");
      event.target.value = "";
      return;
    }

    if (selectedImage?.preview) {
      URL.revokeObjectURL(selectedImage.preview);
    }

    // Cleanly reset previous assessment and active results for the new image
    setQualityAssessment(null);
    setAnalysisResult(null);
    setUploadStatus("");

    setSelectedImage({
      file,
      preview: URL.createObjectURL(file),
    });

    assessSelectedImage(file);
  }

  function removeImage() {
    if (selectedImage?.preview) {
      URL.revokeObjectURL(selectedImage.preview);
    }

    setSelectedImage(null);
    setUploadStatus("");
    setQualityAssessment(null);
    setAnalysisResult(null);
    setIsAssessingQuality(false);

    const input = document.getElementById("skin-image");
    if (input) input.value = "";
  }

  async function uploadImage() {
    if (!selectedImage || isUploading) {
      setUploadStatus("Please select a skin image first.");
      return;
    }

    if (isAssessingQuality) {
      setUploadStatus("Please wait while image quality is being assessed.");
      return;
    }

    if (!qualityAssessment?.is_acceptable) {
      setUploadStatus("Please choose an image that passes the quality assessment before submitting.");
      return;
    }

    if (!affectedBodyPart || !conditionDuration) {
      setUploadStatus("Please select the affected body part and condition duration.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage.file);
    formData.append("affected_body_part", affectedBodyPart);
    formData.append("condition_duration", conditionDuration);
    formData.append("symptoms", JSON.stringify(symptoms));

    setIsUploading(true);
    setUploadStatus("Submitting your case for deep learning inference...");

    try {
      const response = await fetch(`${API_URL}/upload-image`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          getReadableError(data.detail, "Case submission failed.")
        );
      }

      const pred = data.prediction || null;
      setAnalysisResult(pred);
      if (selectedImage?.preview) {
        setLastAnalyzedImage(selectedImage.preview);
      }
      const recipientMail = data.recipient_email || user?.email || "registered email";
      setEmailAlertInfo({
        caseId: data.case_id,
        recipientEmail: recipientMail,
        timestamp: new Date().toLocaleTimeString(),
      });
      setUploadStatus(
        `Case submitted and analyzed successfully! Diagnostic alert sent to ${recipientMail} (Case #${data.case_id ?? "New"}).`
      );

      // Section 17 Notification triggers
      if (pred) {
        addNotification({
          title: "AI Analysis Completed",
          message: `Prediction completed for ${pred.disease_name || pred.predicted_class} with ${pred.confidence}% confidence (Case #${data.case_id || "New"}).`,
          level: "Informational",
          type: "analysis_completed",
        });

        addNotification({
          title: "✉️ Email Report Alert Dispatched",
          message: `Automated diagnostic summary and clinical triage guidance sent directly to ${recipientMail}.`,
          level: "Informational",
          type: "email_sent",
        });

        if (parseFloat(pred.confidence) < 60) {
          addNotification({
            title: "Low-Confidence Prediction Warning",
            message: `Model confidence (${pred.confidence}%) is below 60%. Verification by a certified clinician is strongly recommended.`,
            level: "Medium",
            type: "low_confidence",
          });
        }

        if ((pred.risk_level || "").toLowerCase().includes("high")) {
          addNotification({
            title: "High-Risk Malignancy Alert",
            message: `High risk detected (${pred.disease_name}). Prompt surgical excision / clinical biopsy recommended.`,
            level: "Critical",
            type: "high_risk",
          });
        } else if ((pred.risk_level || "").toLowerCase().includes("moderate")) {
          addNotification({
            title: "Clinical Review Recommended",
            message: `Moderate risk (${pred.disease_name}). Clinical monitoring and examination advised.`,
            level: "High",
            type: "clinical_review",
          });
        }

        addNotification({
          title: "Clinical Report Generated",
          message: `Full diagnostic summary for Case #${data.case_id || "New"} is available to download in PDF, CSV, or JSON.`,
          level: "Informational",
          type: "report_generated",
        });

        addNotification({
          title: "Follow-Up Reminder Scheduled",
          message: `Clinical follow-up scheduled for 30 days to re-evaluate lesion size and evolution.`,
          level: "Low",
          type: "follow_up",
        });
      }

      setSelectedImage(null);
      setAffectedBodyPart("");
      setConditionDuration("");
      setSymptoms([]);

      const deviceInput = document.getElementById("skin-image");
      if (deviceInput) deviceInput.value = "";
    } catch (error) {
      setUploadStatus(
        typeof error?.message === "string"
          ? error.message
          : getReadableError(error, "Case submission failed. Please try again.")
      );
    } finally {
      setIsUploading(false);
    }
  }

  // ---------------- ADMIN CLINICIAN APPROVAL ----------------

  async function loadPendingClinicians() {
    try {
      const response = await fetch(`${API_URL}/admin/clinicians/pending`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load clinician requests.");
      setPendingClinicians(data.pending_clinicians || []);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function approveClinician(clinicianId) {
    try {
      const response = await fetch(
        `${API_URL}/admin/clinicians/${clinicianId}/approve`,
        {
          method: "PATCH",
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not approve clinician.");
      setPendingClinicians((current) =>
        current.filter((clinician) => clinician.user_id !== clinicianId)
      );
      setMessage("Clinician approved successfully.");
    } catch (error) {
      setMessage(error.message);
    }
  }

  // ---------------- DASHBOARD DATA FETCHING ----------------

  async function loadDashboard() {
    if (!token || !user) return;
    setDashboardLoading(true);
    try {
      const endpoint =
        user.role === "clinician" ? "/clinician/dashboard"
        : user.role === "admin"    ? "/admin/dashboard"
        : "/user/dashboard";
      const res = await fetch(`${API_URL}${endpoint}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok) setDashboardData(data);
      else setMessage(data.detail || "Could not load dashboard.");
    } catch {
      setMessage("Network error — could not load dashboard.");
    } finally {
      setDashboardLoading(false);
    }
  }

  async function submitClinicianReview(caseId) {
    try {
      const res = await fetch(`${API_URL}/clinician/cases/${caseId}/review`, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ status: reviewStatus, clinical_notes: reviewNotes }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(data.message || "Review submitted.");
        setSelectedCase(null);
        setReviewNotes("");
        loadDashboard();
      } else {
        setMessage(data.detail || "Review failed.");
      }
    } catch {
      setMessage("Network error — review could not be submitted.");
    }
  }

  async function adminUpdateRole(userId, newRole) {
    try {
      const res = await fetch(`${API_URL}/admin/users/${userId}/role`, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ role: newRole }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(data.message || "Role updated.");
        loadDashboard();
      } else {
        setMessage(data.detail || "Role update failed.");
      }
    } catch {
      setMessage("Network error — role could not be updated.");
    }
  }

  // ---------------- NAVIGATION ----------------

  const userNavigation = [
    "Dashboard",
    "New Analysis",
    "Previous Predictions",
    "Prediction History",
    "Image Results",
    "Reports",
    "Notifications & Alerts",
    "Profile",
  ];

  const clinicianNavigation = [
    "Dashboard",
    "Total Cases",
    "Disease Distribution",
    "Prediction Confidence",
    "Model Performance",
    "Notifications & Alerts",
    "Profile",
  ];

  const adminNavigation = [
    "Dashboard",
    "ML Analytics",
    "Registered Users",
    "Clinician Approvals",
    "Total Analyses",
    "Platform Health",
    "Notifications & Alerts",
    "Profile",
  ];

  const navigation =
    user?.role === "clinician"
      ? clinicianNavigation
      : user?.role === "admin"
      ? adminNavigation
      : userNavigation;

  // ---------------- DASHBOARD DATA ----------------

  const dashboardItems = {
    user: [
      ["Recent analyses", "View your latest image analyses"],
      ["Previous predictions", "Review earlier predictions"],
      ["Prediction history", "Track prediction history"],
      ["Image results", "View uploaded image results"],
      ["Confidence scores", "Review model confidence scores"],
      ["Reports", "Access your analysis reports"],
    ],

    clinician: [
      ["Total Cases", "View, filter and review all clinical cases"],
      ["Disease Distribution", "View disease-wise statistics"],
      ["Prediction Confidence", "Review prediction confidence"],
      ["Model Performance", "View model evaluation"],
    ],

    admin: [
      ["Registered Users", "Manage platform users"],
      ["Clinician Approvals", "Review clinician registrations"],
      ["Total Analyses", "View platform diagnostic volume & distribution"],
      ["Platform Health", "Monitor platform status & metrics"],
    ],
  };

  // ---------------- PUBLIC LANDING PAGE ----------------

  if (page === "landing") {
    return (
      <div className="landing-wrapper">
        {/* Sticky Navigation Bar */}
        <header className="landing-nav">
          <div className="landing-nav-container">
            <div className="landing-brand" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
              <img src="/dermai-icon.svg" alt="DermAI Logo" className="brand-logo-nav" />
              <div className="brand-name-group">
                <span className="brand-name-main">DermAI</span>
                <span className="brand-badge-nav">CLINICAL AI</span>
              </div>
            </div>

            <nav className="landing-nav-links desktop-only">
              <a href="#hero">Home</a>
              <a href="#about">About Project</a>
              <a href="#technology">Technology</a>
              <a href="#dataset">Dataset</a>
              <a href="#diseases">Diseases</a>
              <a href="#how-it-works">How It Works</a>
              <a href="#performance">Performance</a>
            </nav>

            <div className="landing-nav-actions">
              <button
                type="button"
                className="nav-btn-subtle"
                onClick={() => {
                  setMessage("");
                  setPage("role_select");
                }}
              >
                Sign In
              </button>
              <button
                type="button"
                className="nav-btn-primary"
                onClick={() => {
                  setMessage("");
                  setPage("role_select");
                }}
              >
                Get Started <ArrowRight size={15} />
              </button>
              <button
                type="button"
                className="mobile-nav-toggle"
                onClick={() => setMobileNavOpen(!mobileNavOpen)}
                aria-label="Toggle navigation"
              >
                {mobileNavOpen ? <X size={22} /> : <Menu size={22} />}
              </button>
            </div>
          </div>

          {/* Mobile Navigation Drawer */}
          {mobileNavOpen && (
            <div className="mobile-nav-drawer">
              <a href="#hero" onClick={() => setMobileNavOpen(false)}>Home</a>
              <a href="#about" onClick={() => setMobileNavOpen(false)}>About Project</a>
              <a href="#technology" onClick={() => setMobileNavOpen(false)}>Technology</a>
              <a href="#dataset" onClick={() => setMobileNavOpen(false)}>Dataset</a>
              <a href="#diseases" onClick={() => setMobileNavOpen(false)}>Diseases</a>
              <a href="#how-it-works" onClick={() => setMobileNavOpen(false)}>How It Works</a>
              <a href="#performance" onClick={() => setMobileNavOpen(false)}>Performance</a>
              <div className="mobile-drawer-actions">
                <button
                  type="button"
                  className="nav-btn-subtle w-full"
                  onClick={() => {
                    setMobileNavOpen(false);
                    setMessage("");
                    setPage("role_select");
                  }}
                >
                  Sign In to Workspace
                </button>
                <button
                  type="button"
                  className="nav-btn-primary w-full"
                  onClick={() => {
                    setMobileNavOpen(false);
                    setMessage("");
                    setPage("role_select");
                  }}
                >
                  Get Started <ArrowRight size={15} />
                </button>
              </div>
            </div>
          )}
        </header>

        {/* Hero Section */}
        <section id="hero" className="landing-hero-section">
          <div className="landing-hero-container">
            <div className="hero-text-col">
              <div className="hero-pill-badge">
                <Sparkles size={14} className="sparkle-svg" />
                <span>Deep Learning & Explainable AI Screening</span>
              </div>

              <h1 className="hero-main-heading">
                Smarter Skin Analysis.{" "}
                <span className="hero-gradient-span">
                  Powered by Artificial Intelligence.
                </span>
              </h1>

              <p className="hero-subheading-text">
                An AI-powered skin image analysis platform designed to assist preliminary skin disease screening through deep learning and explainable AI.
              </p>

              <div className="hero-cta-group">
                <button
                  type="button"
                  className="btn-hero-solid"
                  onClick={() => {
                    setMessage("");
                    setPage("role_select");
                  }}
                >
                  Get Started <ArrowRight size={18} />
                </button>
                <button
                  type="button"
                  className="btn-hero-outline"
                  onClick={() => {
                    const el = document.getElementById("about");
                    if (el) el.scrollIntoView({ behavior: "smooth" });
                  }}
                >
                  Explore DermAI <ChevronDown size={18} />
                </button>
              </div>

              {/* Trust & Performance Highlights */}
              <div className="hero-trust-bar">
                <div className="trust-item">
                  <strong>98.07%</strong>
                  <span>Train Accuracy</span>
                </div>
                <div className="trust-divider" />
                <div className="trust-item">
                  <strong>97.69%</strong>
                  <span>Melanoma ROC-AUC</span>
                </div>
                <div className="trust-divider" />
                <div className="trust-item">
                  <strong>6 Classes</strong>
                  <span>PAD-UFES-20 Benchmark</span>
                </div>
                <div className="trust-divider" />
                <div className="trust-item">
                  <strong>Grad-CAM</strong>
                  <span>Explainable Heatmaps</span>
                </div>
              </div>
            </div>

            {/* Right Visual AI Scanner Mockup */}
            <div className="hero-visual-col">
              <div className="scanner-glass-card">
                <div className="scanner-header-bar">
                  <div className="traffic-dots">
                    <span className="dot red" />
                    <span className="dot yellow" />
                    <span className="dot green" />
                  </div>
                  <div className="scanner-title-hud">
                    <Activity size={14} className="hud-pulse-icon" />
                    <span>DermAI Multimodal Inference Engine &bull; LIVE</span>
                  </div>
                  <span className="hud-status-badge">PASS</span>
                </div>

                <div className="scanner-viewport-box">
                  <div className="scanner-grid-bg" />
                  <div className="scanner-laser-beam" />
                  
                  {/* Lesion Silhouette / Visual Simulation */}
                  <div className="scanner-lesion-sim">
                    <div className="sim-lesion-body" />
                    <div className="sim-bounding-box">
                      <span className="corner top-left" />
                      <span className="corner top-right" />
                      <span className="corner bottom-left" />
                      <span className="corner bottom-right" />
                      <span className="box-label">ROI: 99.2% Conf</span>
                    </div>
                  </div>

                  {/* Realtime Detection Banner */}
                  <div className="scanner-detection-banner">
                    <div className="detect-col">
                      <span className="detect-caption">PREDICTED CLASS</span>
                      <strong className="detect-name">Melanocytic Nevus (NEV)</strong>
                    </div>
                    <div className="detect-badge-pill green">
                      <span>Low Risk &bull; Benign</span>
                    </div>
                  </div>
                </div>

                <div className="scanner-hud-footer">
                  <div className="hud-metric">
                    <span className="hud-label">Image Sharpness</span>
                    <strong className="hud-val text-teal">Passed (142.6)</strong>
                  </div>
                  <div className="hud-metric">
                    <span className="hud-label">Architecture</span>
                    <strong className="hud-val">EfficientNet-B0</strong>
                  </div>
                  <div className="hud-metric">
                    <span className="hud-label">Inference Time</span>
                    <strong className="hud-val">34.09 ms</strong>
                  </div>
                  <div className="hud-metric">
                    <span className="hud-label">Attention Layer</span>
                    <strong className="hud-val text-teal">Grad-CAM MBConv6</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* About DermAI Section */}
        <section id="about" className="landing-section-wrap bg-light">
          <div className="landing-container">
            <div className="section-header-center">
              <span className="section-eyebrow">ABOUT DERMAI</span>
              <h2 className="section-title">Bridging Mobile Imaging and Dermatological Decision Support</h2>
              <p className="section-desc">
                DermAI is a deep learning-based skin disease analysis platform that analyzes uploaded or captured skin images and provides predicted classifications alongside explainable AI visualizations. It leverages multimodal clinical feature fusion and pre-inference image quality gating to ensure diagnostic accuracy and clinical reliability.
              </p>
            </div>

            <div className="about-cards-grid">
              <div className="about-feature-card">
                <div className="feature-icon-wrapper teal">
                  <BrainCircuit size={28} />
                </div>
                <h3>AI-Powered Multimodal Analysis</h3>
                <p>
                  Combines EfficientNet-B0 visual representations with 64 patient clinical metadata dimensions (age, anatomical site, lesion itch, bleeding, elevation, and growth) for holistic diagnostic reasoning.
                </p>
                <ul className="feature-bullet-list">
                  <li><Check size={16} /> 1,280 deep CNN visual features</li>
                  <li><Check size={16} /> Patient-reported clinical risk indicators</li>
                  <li><Check size={16} /> Calibrated probability outputs across 6 classes</li>
                </ul>
              </div>

              <div className="about-feature-card">
                <div className="feature-icon-wrapper blue">
                  <Eye size={28} />
                </div>
                <h3>Image Quality Pre-Assessment</h3>
                <p>
                  Real-time OpenCV validation pipeline testing sharpness (Laplacian variance), illumination range, and contrast before AI inference to prevent misdiagnosis.
                </p>
                <ul className="feature-bullet-list">
                  <li><Check size={16} /> Blurriness & motion blur detection</li>
                  <li><Check size={16} /> Low-light and over-exposure filtering</li>
                  <li><Check size={16} /> Immediate guidance for photo retake</li>
                </ul>
              </div>

              <div className="about-feature-card">
                <div className="feature-icon-wrapper indigo">
                  <Sparkles size={28} />
                </div>
                <h3>Explainable AI (Grad-CAM)</h3>
                <p>
                  Transparent visual attention heatmaps pinpointing the exact lesion pixels and morphological borders that drove the neural network's classification.
                </p>
                <ul className="feature-bullet-list">
                  <li><Check size={16} /> High-resolution saliency overlays</li>
                  <li><Check size={16} /> Clinically interpretable lesion boundaries</li>
                  <li><Check size={16} /> Eliminates black-box AI opacity</li>
                </ul>
              </div>

              <div className="about-feature-card">
                <div className="feature-icon-wrapper emerald">
                  <Stethoscope size={28} />
                </div>
                <h3>Clinical Decision Support</h3>
                <p>
                  Prioritized triage categorization (Low, Moderate, High, Urgent), evidence-based dermatological clinical protocols, and automated report delivery.
                </p>
                <ul className="feature-bullet-list">
                  <li><Check size={16} /> Risk-stratified triage guidance</li>
                  <li><Check size={16} /> Exportable signed PDF clinical reports</li>
                  <li><Check size={16} /> Automated clinician alert notifications</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Dataset Section */}
        <section id="dataset" className="landing-section-wrap">
          <div className="landing-container">
            <div className="section-header-center">
              <span className="section-eyebrow">CLINICAL DATASET</span>
              <h2 className="section-title">Trained on the PAD-UFES-20 Dermatological Benchmark</h2>
              <p className="section-desc">
                Collected by the Federal University of Espírito Santo (UFES), PAD-UFES-20 reflects real-world clinical conditions. Unlike sanitized dermoscopic datasets, it uses smartphone cameras across diverse patient demographics, providing robust real-world generalization.
              </p>
            </div>

            <div className="dataset-stats-strip">
              <div className="dataset-stat-box">
                <span className="dataset-stat-number">2,298</span>
                <span className="dataset-stat-label">Clinical Smartphone Images</span>
                <span className="dataset-stat-sub">High-resolution mobile captures</span>
              </div>
              <div className="dataset-stat-box">
                <span className="dataset-stat-number">1,373</span>
                <span className="dataset-stat-label">Unique Patients</span>
                <span className="dataset-stat-sub">Real dermatological clinic visits</span>
              </div>
              <div className="dataset-stat-box">
                <span className="dataset-stat-number">6</span>
                <span className="dataset-stat-label">Disease Categories</span>
                <span className="dataset-stat-sub">Malignant, premalignant & benign</span>
              </div>
              <div className="dataset-stat-box">
                <span className="dataset-stat-number">Patient-Wise</span>
                <span className="dataset-stat-label">Grouped Partitioning</span>
                <span className="dataset-stat-sub">Zero data leakage across train/val/test</span>
              </div>
            </div>
          </div>
        </section>

        {/* Diseases Section */}
        <section id="diseases" className="landing-section-wrap bg-light">
          <div className="landing-container">
            <div className="section-header-center">
              <span className="section-eyebrow">PATHOLOGY TAXONOMY</span>
              <h2 className="section-title">6 Skin Diseases Analyzed by DermAI</h2>
              <p className="section-desc">
                Click any condition to view detailed clinical characteristics, typical morphological features, malignancy risk, and recommended triage protocol.
              </p>
            </div>

            <div className="diseases-grid">
              {Object.entries(DISEASE_DETAILS).map(([code, dis]) => (
                <div
                  key={code}
                  className="disease-card"
                  onClick={() => setActiveDiseaseModal(dis)}
                >
                  <div className="disease-card-header">
                    <span className="disease-code-pill" style={{ borderColor: dis.riskColor, color: dis.riskColor }}>
                      {dis.code}
                    </span>
                    <span className="disease-risk-tag" style={{ backgroundColor: `${dis.riskColor}18`, color: dis.riskColor }}>
                      {dis.risk}
                    </span>
                  </div>

                  <h3 className="disease-title">{dis.name}</h3>
                  <p className="disease-type-sub">{dis.type}</p>
                  <p className="disease-brief-desc">{dis.description.substring(0, 115)}...</p>

                  <div className="disease-card-footer">
                    <span className="dataset-freq-badge">{dis.datasetCount}</span>
                    <button type="button" className="disease-detail-link">
                      View Details &rarr;
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Interactive Disease Modal */}
        {activeDiseaseModal && (
          <div className="modal-backdrop-blur" onClick={() => setActiveDiseaseModal(null)}>
            <div className="disease-modal-card" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header-row">
                <div className="modal-title-wrap">
                  <div className="modal-code-badge" style={{ backgroundColor: `${activeDiseaseModal.riskColor}20`, color: activeDiseaseModal.riskColor }}>
                    {activeDiseaseModal.code}
                  </div>
                  <div>
                    <h2 className="modal-disease-name">{activeDiseaseModal.name}</h2>
                    <span className="modal-disease-type">{activeDiseaseModal.type}</span>
                  </div>
                </div>
                <button type="button" className="modal-close-x" onClick={() => setActiveDiseaseModal(null)}>
                  <X size={20} />
                </button>
              </div>

              <div className="modal-body-scroll">
                <div className="modal-risk-banner" style={{ borderLeftColor: activeDiseaseModal.riskColor }}>
                  <AlertTriangle size={18} style={{ color: activeDiseaseModal.riskColor }} />
                  <div>
                    <strong>Malignancy Classification: {activeDiseaseModal.risk}</strong>
                    <p>Dataset Representation: {activeDiseaseModal.datasetCount}</p>
                  </div>
                </div>

                <div className="modal-desc-box">
                  <h4>Clinical Overview</h4>
                  <p>{activeDiseaseModal.description}</p>
                </div>

                <div className="modal-features-box">
                  <h4>Diagnostic Hallmarks & Morphological Criteria</h4>
                  <ul>
                    {activeDiseaseModal.keyFeatures.map((feat, idx) => (
                      <li key={idx}><Check size={16} /> <span>{feat}</span></li>
                    ))}
                  </ul>
                </div>

                <div className="modal-triage-box">
                  <h4>Recommended Triage Protocol</h4>
                  <p>{activeDiseaseModal.triageAction}</p>
                </div>
              </div>

              <div className="modal-footer-row">
                <button type="button" className="btn-modal-close" onClick={() => setActiveDiseaseModal(null)}>
                  Close Overview
                </button>
                <button
                  type="button"
                  className="btn-modal-screen"
                  onClick={() => {
                    setActiveDiseaseModal(null);
                    setPage("role_select");
                  }}
                >
                  Start Screening Now &rarr;
                </button>
              </div>
            </div>
          </div>
        )}

        {/* How It Works Section */}
        <section id="how-it-works" className="landing-section-wrap">
          <div className="landing-container">
            <div className="section-header-center">
              <span className="section-eyebrow">WORKFLOW</span>
              <h2 className="section-title">How DermAI Works</h2>
              <p className="section-desc">
                From mobile photograph capture to clinical decision support in 6 transparent, verified steps.
              </p>
            </div>

            <div className="workflow-steps-grid">
              {WORKFLOW_STEPS.map((wf, idx) => {
                const IconComponent = wf.icon;
                return (
                  <div key={idx} className="workflow-card">
                    <div className="workflow-card-top">
                      <span className="workflow-step-num">{wf.step}</span>
                      <div className="workflow-icon-box">
                        <IconComponent size={22} />
                      </div>
                    </div>
                    <h3>{wf.title}</h3>
                    <p>{wf.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Technology Stack Section */}
        <section id="technology" className="landing-section-wrap bg-light">
          <div className="landing-container">
            <div className="section-header-center">
              <span className="section-eyebrow">ARCHITECTURE</span>
              <h2 className="section-title">State-of-the-Art Technology Stack</h2>
              <p className="section-desc">
                Engineered for speed, clinical accuracy, model explainability, and enterprise-grade medical data security.
              </p>
            </div>

            <div className="tech-stack-grid">
              {TECH_STACK.map((item, idx) => {
                const TechIcon = item.icon;
                return (
                  <div key={idx} className="tech-item-card">
                    <div className="tech-icon-wrap">
                      <TechIcon size={24} />
                    </div>
                    <div>
                      <span className="tech-role-badge">{item.role}</span>
                      <h4 className="tech-name">{item.name}</h4>
                      <p className="tech-desc">{item.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Model Performance Section */}
        <section id="performance" className="landing-section-wrap">
          <div className="landing-container">
            <div className="section-header-center">
              <span className="section-eyebrow">EVALUATION BENCHMARK</span>
              <h2 className="section-title">Verified Model Performance</h2>
              <p className="section-desc">
                Quantitative performance metrics evaluated on the held-out test split of PAD-UFES-20.
              </p>
            </div>

            <div className="metrics-cards-grid">
              {MODEL_METRICS.map((met, idx) => (
                <div key={idx} className="metric-score-card">
                  <span className="metric-score-label">{met.label}</span>
                  <span className="metric-score-value" style={{ color: met.color }}>{met.value}</span>
                  <span className="metric-score-note">{met.note}</span>
                </div>
              ))}
            </div>

            <div className="architecture-justification-banner">
              <div className="justification-header">
                <Award size={20} className="award-icon" />
                <h4>Production Selection: EfficientNet-B0 Compound Scaling</h4>
              </div>
              <p>
                While ResNet50 achieved 77.83% accuracy, its recall on malignant classes was only 64.13%. EfficientNet-B0 was selected for production because of its superior <strong>70.47% Macro Recall</strong>, <strong>97.69% Melanoma ROC-AUC</strong>, and rapid <strong>34.09 ms inference latency</strong>, making it the most clinically sensitive model for urgent malignant lesions.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer id="contact" className="landing-footer">
          <div className="landing-container">
            <div className="footer-top-grid">
              <div className="footer-brand-col">
                <div className="footer-brand-row">
                  <img src="/dermai-icon.svg" alt="DermAI Icon" className="footer-logo-svg" />
                  <span className="footer-brand-title">DermAI</span>
                </div>
                <p className="footer-mission-text">
                  Advanced multimodal skin disease screening powered by compound-scaled deep learning and explainable AI.
                </p>
                <div className="footer-compliance-badges">
                  <span>PAD-UFES-20 Benchmarked</span>
                  <span>ISO-Aligned Architecture</span>
                </div>
              </div>

              <div className="footer-nav-col">
                <h4>Navigation</h4>
                <a href="#hero">Home</a>
                <a href="#about">About Project</a>
                <a href="#technology">Technology</a>
                <a href="#dataset">Dataset</a>
                <a href="#diseases">Diseases</a>
                <a href="#how-it-works">How It Works</a>
                <a href="#performance">Performance</a>
              </div>

              <div className="footer-portals-col">
                <h4>Clinical Portals</h4>
                <button type="button" onClick={() => { setMessage(""); setPage("login_patient"); }}>Patient Portal</button>
                <button type="button" onClick={() => { setMessage(""); setPage("login_clinician"); }}>Clinician Portal</button>
                <button type="button" onClick={() => { setMessage(""); setPage("login_admin"); }}>Administrator Portal</button>
                <button type="button" onClick={() => { setMessage(""); setPage("role_select"); }}>Portal Selection</button>
              </div>

              <div className="footer-contact-col">
                <h4>Platform & Support</h4>
                <p>Support: <a href="mailto:dermai.diagnostics@gmail.com">dermai.diagnostics@gmail.com</a></p>
                <p>Deployment: Docker Container on Render Cloud</p>
                <p>API Status: Operational (PostgreSQL + PyTorch)</p>
              </div>
            </div>

            {/* Medical Disclaimer Banner */}
            <div className="footer-medical-disclaimer">
              <div className="disclaimer-header">
                <AlertTriangle size={18} />
                <strong>CLINICAL & REGULATORY MEDICAL DISCLAIMER</strong>
              </div>
              <p>
                DermAI is an artificial intelligence-assisted diagnostic screening tool intended for preliminary screening, educational research, and clinical decision support. DermAI is not an autonomous medical device and does not provide definitive medical diagnoses. All outputs, including predictions, risk stratifications, and Grad-CAM saliency visualizations, must be verified by a licensed healthcare professional, dermatologist, or pathologist before initiating clinical interventions.
              </p>
            </div>

            <div className="footer-bottom-row">
              <p>&copy; {new Date().getFullYear()} DermAI Platform. All rights reserved.</p>
              <div className="footer-links-row">
                <span>Privacy Protocol</span>
                <span>&bull;</span>
                <span>Terms of Medical Use</span>
                <span>&bull;</span>
                <span>Model Card & Benchmarks</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    );
  }

  // ---------------- ROLE SELECTION PORTAL ----------------

  if (page === "role_select") {
    return (
      <main className="role-portal-page">
        <div className="role-portal-container">
          <div className="role-portal-header">
            <div className="landing-brand-center" onClick={() => setPage("landing")}>
              <img src="/dermai-icon.svg" alt="DermAI" className="role-logo-img" />
              <div className="brand-name-group">
                <span className="brand-name-main">DermAI</span>
                <span className="brand-badge-nav">CLINICAL SUITE</span>
              </div>
            </div>

            <h1 className="role-portal-title">Select Your Workspace Portal</h1>
            <p className="role-portal-sub">
              Choose the portal tailored to your role in the dermatological screening workflow:
            </p>
          </div>

          <div className="role-cards-container">
            {/* Patient Portal Card */}
            <div
              className="role-selection-card patient"
              onClick={() => {
                setMessage("");
                setPage("login_patient");
              }}
            >
              <div className="role-card-badge patient">Patient Care</div>
              <div className="role-icon-box patient">
                <HeartPulse size={36} />
              </div>
              <h2 className="role-title">Patient Portal</h2>
              <p className="role-desc">
                Submit smartphone photographs of skin lesions, receive immediate automated quality validation, view preliminary AI predictions, and track your screening records.
              </p>
              <ul className="role-feature-bullets">
                <li><Check size={16} /> Instant photo upload & quality check</li>
                <li><Check size={16} /> Multimodal AI screening results</li>
                <li><Check size={16} /> Personal dermatological history records</li>
              </ul>
              <button
                type="button"
                className="btn-enter-portal patient"
                onClick={(e) => {
                  e.stopPropagation();
                  setMessage("");
                  setPage("login_patient");
                }}
              >
                Enter Patient Portal &rarr;
              </button>
            </div>

            {/* Clinician Portal Card */}
            <div
              className="role-selection-card clinician"
              onClick={() => {
                setMessage("");
                setPage("login_clinician");
              }}
            >
              <div className="role-card-badge clinician">Medical Professional</div>
              <div className="role-icon-box clinician">
                <Stethoscope size={36} />
              </div>
              <h2 className="role-title">Clinician Portal</h2>
              <p className="role-desc">
                Access patient triage queues, examine dual-view Grad-CAM explainability heatmaps, verify diagnostic findings, submit clinical notes, and export verified PDF reports.
              </p>
              <ul className="role-feature-bullets">
                <li><Check size={16} /> Risk-prioritized triage queue</li>
                <li><Check size={16} /> Grad-CAM attention heatmaps</li>
                <li><Check size={16} /> One-click review approval & email alerts</li>
              </ul>
              <button
                type="button"
                className="btn-enter-portal clinician"
                onClick={(e) => {
                  e.stopPropagation();
                  setMessage("");
                  setPage("login_clinician");
                }}
              >
                Enter Clinician Portal &rarr;
              </button>
            </div>

            {/* Administrator Portal Card */}
            <div
              className="role-selection-card admin"
              onClick={() => {
                setMessage("");
                setPage("login_admin");
              }}
            >
              <div className="role-card-badge admin">System Governance</div>
              <div className="role-icon-box admin">
                <Shield size={36} />
              </div>
              <h2 className="role-title">Administrator Portal</h2>
              <p className="role-desc">
                Oversee medical staff credentials, approve new clinician registrations, monitor database connection telemetry, and inspect real-time AI inference performance.
              </p>
              <ul className="role-feature-bullets">
                <li><Check size={16} /> Clinician approval & credentialing</li>
                <li><Check size={16} /> Database audit logs & user management</li>
                <li><Check size={16} /> Model latency & memory telemetry</li>
              </ul>
              <button
                type="button"
                className="btn-enter-portal admin"
                onClick={(e) => {
                  e.stopPropagation();
                  setMessage("");
                  setPage("login_admin");
                }}
              >
                Enter Admin Portal &rarr;
              </button>
            </div>
          </div>

          <div className="role-portal-bottom-action">
            <button
              type="button"
              className="btn-back-home"
              onClick={() => {
                setMessage("");
                setPage("landing");
              }}
            >
              &larr; Return to Public Landing Page
            </button>
          </div>
        </div>
      </main>
    );
  }

  // ---------------- DEDICATED ROLE LOGIN PAGES ----------------

  if (page === "login_patient" || page === "login_clinician" || page === "login_admin" || page === "login") {
    const portalRole =
      page === "login_clinician" ? "clinician" :
      page === "login_admin" ? "admin" : "patient";

    const portalConfig = {
      patient: {
        title: "Patient Portal",
        badge: "Patient Self-Screening Workspace",
        subtitle: "Sign in to access your screening history and submit new lesion images",
        heroTagline: "Personal Dermatological Care & Screening",
        heroDesc: "Submit skin photographs directly from your smartphone, review automated preliminary diagnostic assessments, and securely track your personal health history.",
        bullets: [
          "Instant image quality validation (sharpness & illumination)",
          "Multimodal AI classification across 6 lesion types",
          "Grad-CAM explainable attention heatmap visualizations",
          "Encrypted medical history and clinical tracking"
        ],
        switchText: "Don't have a patient account?",
        switchAction: "Create Patient Account &rarr;",
        onSwitch: () => {
          setRole("user");
          setMessage("");
          setPage("register");
        }
      },
      clinician: {
        title: "Clinician Portal",
        badge: "Diagnostic Review & Decision Support",
        subtitle: "Sign in to examine triage queues and approve diagnostic reports",
        heroTagline: "Clinical Decision Support & Verification",
        heroDesc: "Access prioritized patient screening queues, examine Grad-CAM explainability saliency maps, verify preliminary diagnoses, and dispatch authenticated clinical reports.",
        bullets: [
          "Prioritized triage queue by malignancy risk level",
          "Dual-view Grad-CAM heatmaps & overlay comparisons",
          "PAD-UFES-20 model performance telemetry & benchmarks",
          "One-click clinical report verification and email alert dispatch"
        ],
        switchText: "New medical clinician?",
        switchAction: "Register for Staff Credentialing &rarr;",
        onSwitch: () => {
          setRole("clinician");
          setMessage("");
          setPage("register");
        }
      },
      admin: {
        title: "Administrator Portal",
        badge: "Platform Governance & Security",
        subtitle: "Sign in to manage clinician credentials and system audits",
        heroTagline: "Platform Security & Diagnostic Telemetry",
        heroDesc: "Oversee platform health, approve medical staff accounts, monitor model inference latency, and inspect database audit logs with strict role-based controls.",
        bullets: [
          "Clinician credentialing and account approval workflow",
          "Live PostgreSQL database connection telemetry",
          "Model latency tracking & hardware allocation metrics",
          "Role-based access control and session encryption"
        ],
        switchText: "Need administrative assistance?",
        switchAction: "Contact Technical Operations",
        onSwitch: () => {
          setMessage("Default Administrator credentials: admin@dermai.com / Admin@123");
        }
      }
    }[portalRole];

    return (
      <main className="dedicated-login-page">
        <div className="login-split-container">
          {/* Left Hero Column */}
          <section className={`login-hero-col ${portalRole}`}>
            <div className="login-hero-content">
              <div className="login-hero-brand" onClick={() => setPage("landing")}>
                <img src="/dermai-icon.svg" alt="DermAI" className="hero-brand-svg" />
                <div className="brand-name-group">
                  <span className="brand-name-main text-white">DermAI</span>
                  <span className="brand-badge-hero">{portalRole.toUpperCase()}</span>
                </div>
              </div>

              <div className="login-hero-text-block">
                <span className="login-hero-pill">{portalConfig.badge}</span>
                <h2>{portalConfig.heroTagline}</h2>
                <p>{portalConfig.heroDesc}</p>
              </div>

              <div className="login-hero-bullets">
                {portalConfig.bullets.map((b, i) => (
                  <div key={i} className="hero-bullet-row">
                    <CheckCircle2 size={18} className="bullet-check" />
                    <span>{b}</span>
                  </div>
                ))}
              </div>

              <div className="login-security-tag">
                <Lock size={15} />
                <span>Encrypted Patient Data & ISO-Aligned Inference</span>
              </div>
            </div>
          </section>

          {/* Right Form Column */}
          <section className="login-form-col">
            <div className="login-form-card">
              <div className="form-card-header">
                <div className="form-brand-row">
                  <img src="/dermai-icon.svg" alt="DermAI" className="form-logo-svg" />
                  <span className="form-brand-name">DermAI</span>
                </div>
                <h1 className="form-title">{portalConfig.title}</h1>
                <p className="form-sub">{portalConfig.subtitle}</p>
              </div>

              <form onSubmit={(e) => handleLogin(e, portalRole)} className="login-actual-form">
                <div className="form-group-custom">
                  <label>Email Address</label>
                  <input
                    type="email"
                    placeholder="name@dermai.clinical"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>

                <div className="form-group-custom">
                  <div className="label-action-row">
                    <label>Password</label>
                    <button
                      type="button"
                      className="link-forgot-btn"
                      onClick={() => {
                        setMessage("");
                        setPage("forgot");
                      }}
                    >
                      Forgot password?
                    </button>
                  </div>

                  <div className="password-input-wrapper">
                    <input
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                    <button
                      type="button"
                      className="btn-toggle-eye"
                      onClick={() => setShowPassword(!showPassword)}
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                </div>

                <div className="form-options-row">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={rememberMe}
                      onChange={(e) => setRememberMe(e.target.checked)}
                    />
                    <span>Remember my workspace</span>
                  </label>
                </div>

                <button type="submit" className="btn-submit-login" disabled={loading}>
                  {loading ? (
                    <span className="btn-loading-flex">
                      <RefreshCw size={18} className="spin-icon" />
                      Authenticating...
                    </span>
                  ) : (
                    `Sign In to ${portalConfig.title} →`
                  )}
                </button>
              </form>

              {message && (
                <div className="form-alert-message" role="alert">
                  <Info size={16} />
                  <span>{message}</span>
                </div>
              )}

              <div className="form-card-footer">
                <div className="auth-switch-box">
                  <span>{portalConfig.switchText}</span>{" "}
                  <button type="button" className="link-highlight" onClick={portalConfig.onSwitch}>
                    {portalConfig.switchAction}
                  </button>
                </div>

                <div className="switch-portal-actions">
                  <button
                    type="button"
                    className="btn-subtle-switch"
                    onClick={() => {
                      setMessage("");
                      setPage("role_select");
                    }}
                  >
                    &larr; Switch to Another Portal
                  </button>
                  <button
                    type="button"
                    className="btn-subtle-switch"
                    onClick={() => {
                      setMessage("");
                      setPage("landing");
                    }}
                  >
                    Return to Public Home &rarr;
                  </button>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    );
  }

  // ---------------- REGISTRATION PAGE ----------------

  if (page === "register") {
    return (
      <main className="register-page-wrap">
        <div className="register-container-split">
          {/* Left Hero Column */}
          <section className="register-hero-side">
            <div>
              <div className="hero-status-tag">
                <span className="live-pulse"></span>
                <span>Clinical Multimodal AI Platform</span>
              </div>

              <div className="register-hero-brand" onClick={() => setPage("landing")}>
                <img src="/dermai-icon.svg" alt="DermAI" className="hero-brand-svg" />
                <div>
                  <h2>DermAI Clinical Platform</h2>
                  <span className="hero-tagline">Multimodal Deep Learning Dermatological Screening</span>
                </div>
              </div>

              <div className="register-hero-intro">
                <h3>Empowering Skin Disease Pre-Screening</h3>
                <p>
                  Join DermAI to leverage clinical multimodal AI trained on the PAD-UFES-20 smartphone dataset.
                  Screen and triage skin lesions in real-time with explainable AI support.
                </p>
              </div>

              <div className="hero-metrics-grid">
                <div className="hero-metric-box">
                  <strong>98.07%</strong>
                  <span>Train Accuracy</span>
                </div>
                <div className="hero-metric-box">
                  <strong>76.74%</strong>
                  <span>Test Accuracy</span>
                </div>
                <div className="hero-metric-box">
                  <strong>97.69%</strong>
                  <span>Melanoma AUC</span>
                </div>
                <div className="hero-metric-box">
                  <strong>34 ms</strong>
                  <span>Inference Speed</span>
                </div>
              </div>

              <div className="hero-features-list">
                <div className="hero-feature-item">
                  <div className="feature-dot"></div>
                  <div>
                    <h4>Real-time Image Quality Assessment</h4>
                    <p>OpenCV validation checks sharpness & illumination before clinical inference.</p>
                  </div>
                </div>

                <div className="hero-feature-item">
                  <div className="feature-dot"></div>
                  <div>
                    <h4>Multimodal Fusion Architecture</h4>
                    <p>Integrates 64 clinical features with EfficientNet-B0 visual representations.</p>
                  </div>
                </div>

                <div className="hero-feature-item">
                  <div className="feature-dot"></div>
                  <div>
                    <h4>Explainable AI with Grad-CAM</h4>
                    <p>Visual attention heatmaps provide transparent decision-making support.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="register-hero-footer">
              <p>🔒 End-to-end encrypted medical data &middot; HIPAA aligned cloud deployment</p>
            </div>
          </section>

          {/* Right Form Column */}
          <section className="register-form-side">
            <div className="register-card">
              <div className="form-card-header">
                <div className="form-brand-row">
                  <img src="/dermai-icon.svg" alt="DermAI" className="form-logo-svg" />
                  <span className="form-brand-name">DermAI</span>
                </div>
                <h2>Create an Account</h2>
                <p className="subtitle">
                  Join the platform to screen, review, or manage dermatological diagnostic cases.
                </p>
              </div>

              {/* Role Toggle Selector */}
              <div className="role-selector-wrap">
                <label className="field-label">I am registering as:</label>
                <div className="role-toggle-group">
                  <button
                    type="button"
                    className={`role-toggle-btn ${role === "user" ? "active" : ""}`}
                    onClick={() => setRole("user")}
                  >
                    <span className="role-btn-icon">👤</span>
                    <div>
                      <strong>Patient</strong>
                      <small>Upload & monitor cases</small>
                    </div>
                  </button>

                  <button
                    type="button"
                    className={`role-toggle-btn ${role === "clinician" ? "active" : ""}`}
                    onClick={() => setRole("clinician")}
                  >
                    <span className="role-btn-icon">🩺</span>
                    <div>
                      <strong>Clinician</strong>
                      <small>Diagnostic review & notes</small>
                    </div>
                  </button>
                </div>

                {role === "clinician" && (
                  <div className="clinician-notice-banner">
                    <span className="notice-icon">ℹ️</span>
                    <p>Clinician accounts require administrator credential verification before accessing patient records.</p>
                  </div>
                )}
              </div>

              <form onSubmit={handleRegister} className="register-form">
                {/* Account Details */}
                <div className="form-section-title">Account Credentials</div>

                <div className="form-row-2col">
                  <div className="form-group-login">
                    <label>Full Name *</label>
                    <input
                      type="text"
                      placeholder={role === "clinician" ? "Dr. Jane Doe" : "Jane Doe"}
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                    />
                  </div>

                  <div className="form-group-login">
                    <label>Email Address *</label>
                    <input
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="form-row-2col">
                  <div className="form-group-login">
                    <label>Password (min 8 chars) *</label>
                    <div className="password-input-wrapper">
                      <input
                        type={showPassword ? "text" : "password"}
                        placeholder="At least 8 characters"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        minLength={8}
                        required
                      />
                      <button
                        type="button"
                        className="btn-toggle-eye"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>

                  <div className="form-group-login">
                    <label>Confirm Password *</label>
                    <div className="password-input-wrapper">
                      <input
                        type={showConfirmPassword ? "text" : "password"}
                        placeholder="Re-enter password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        minLength={8}
                        required
                      />
                      <button
                        type="button"
                        className="btn-toggle-eye"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      >
                        {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Patient Profile Fields */}
                {role === "user" && (
                  <div className="patient-fields-block">
                    <div className="form-section-title">Patient Profile Information</div>

                    <div className="form-row-2col">
                      <div className="form-group-login">
                        <label>Age *</label>
                        <input
                          type="number"
                          placeholder="e.g. 45"
                          value={patientAge}
                          onChange={(e) => setPatientAge(e.target.value)}
                          min={1}
                          max={120}
                          required
                        />
                      </div>

                      <div className="form-group-login">
                        <label>Biological Sex</label>
                        <select
                          value={patientGender}
                          onChange={(e) => setPatientGender(e.target.value)}
                        >
                          <option value="">Select sex</option>
                          <option value="female">Female</option>
                          <option value="male">Male</option>
                          <option value="other">Other</option>
                        </select>
                      </div>
                    </div>

                    <div className="form-group-login">
                      <label>Medical History</label>
                      <textarea
                        rows={2}
                        placeholder="Relevant medical history (e.g. sun exposure, family history of melanoma)"
                        value={medicalHistory}
                        onChange={(e) => setMedicalHistory(e.target.value)}
                      />
                    </div>

                    <div className="form-row-2col">
                      <div className="form-group-login">
                        <label>Previous Skin Treatments</label>
                        <input
                          type="text"
                          placeholder="e.g. Cryotherapy, excision, none"
                          value={previousSkinTreatments}
                          onChange={(e) => setPreviousSkinTreatments(e.target.value)}
                        />
                      </div>

                      <div className="form-group-login">
                        <label>Current Medications</label>
                        <input
                          type="text"
                          placeholder="e.g. Immunosuppressants, none"
                          value={currentMedications}
                          onChange={(e) => setCurrentMedications(e.target.value)}
                        />
                      </div>
                    </div>
                  </div>
                )}

                <button type="submit" className="btn-login-submit" disabled={loading}>
                  {loading
                    ? "Creating Account..."
                    : role === "clinician"
                    ? "Submit Clinician Application &rarr;"
                    : "Complete Registration &rarr;"}
                </button>
              </form>

              {message && (
                <div className="form-alert-message" role="alert">
                  <Info size={16} />
                  <span>{message}</span>
                </div>
              )}

              <div className="login-footer">
                <p className="auth-switch">
                  Already have an account?{" "}
                  <button
                    type="button"
                    className="text-button-highlight"
                    onClick={() => {
                      setMessage("");
                      setPage("role_select");
                    }}
                  >
                    Sign In &rarr;
                  </button>
                </p>
                <button
                  type="button"
                  className="btn-subtle-switch"
                  onClick={() => {
                    setMessage("");
                    setPage("landing");
                  }}
                >
                  &larr; Return to Public Home
                </button>
              </div>
            </div>
          </section>
        </div>
      </main>
    );
  }

  // ---------------- FORGOT PASSWORD PAGE ----------------

  if (page === "forgot") {
    return (
      <main className="dedicated-login-page">
        <div className="login-split-container single-center">
          <div className="login-form-card center-card">
            <div className="form-card-header">
              <div className="form-brand-row">
                <img src="/dermai-icon.svg" alt="DermAI" className="form-logo-svg" />
                <span className="form-brand-name">DermAI</span>
              </div>
              <h1 className="form-title">Reset Your Password</h1>
              <p className="form-sub">
                Enter your registered email address below. We'll send you a secure link to reset your password.
              </p>
            </div>

            <form onSubmit={handleForgotPassword} className="login-actual-form">
              <div className="form-group-custom">
                <label>Registered Email Address</label>
                <input
                  type="email"
                  placeholder="name@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <button type="submit" className="btn-submit-login" disabled={loading}>
                {loading ? (
                  <span className="btn-loading-flex">
                    <RefreshCw size={18} className="spin-icon" />
                    Dispatching Link...
                  </span>
                ) : (
                  "Send Password Reset Link →"
                )}
              </button>
            </form>

            {message && (
              <div className="form-alert-message" role="alert">
                <Info size={16} />
                <span>{message}</span>
              </div>
            )}

            <div className="form-card-footer text-center">
              <button
                type="button"
                className="btn-subtle-switch"
                onClick={() => {
                  setMessage("");
                  setPage("role_select");
                }}
              >
                &larr; Back to Workspace Portals
              </button>
            </div>
          </div>
        </div>
      </main>
    );
  }

  // ---------------- RESET PASSWORD PAGE ----------------

  if (page === "reset_password") {
    return (
      <main className="dedicated-login-page">
        <div className="login-split-container single-center">
          <div className="login-form-card center-card">
            <div className="form-card-header">
              <div className="form-brand-row">
                <img src="/dermai-icon.svg" alt="DermAI" className="form-logo-svg" />
                <span className="form-brand-name">DermAI</span>
              </div>
              <h1 className="form-title">Set New Password</h1>
              <p className="form-sub">
                Enter your new secure password for your DermAI account (minimum 6 characters).
              </p>
            </div>

            <form onSubmit={handleResetPassword} className="login-actual-form">
              {!resetToken && (
                <div className="form-group-custom">
                  <label>Password Reset Token</label>
                  <input
                    type="text"
                    placeholder="Paste reset token here"
                    value={resetToken}
                    onChange={(e) => setResetToken(e.target.value)}
                    required
                  />
                </div>
              )}

              <div className="form-group-custom">
                <label>New Password</label>
                <div className="password-input-wrapper">
                  <input
                    type={showResetPassword ? "text" : "password"}
                    placeholder="Enter new password (min 6 characters)"
                    value={resetNewPassword}
                    onChange={(e) => setResetNewPassword(e.target.value)}
                    minLength={6}
                    required
                  />
                  <button
                    type="button"
                    className="btn-toggle-eye"
                    onClick={() => setShowResetPassword(!showResetPassword)}
                  >
                    {showResetPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              <div className="form-group-custom">
                <label>Confirm New Password</label>
                <div className="password-input-wrapper">
                  <input
                    type={showResetPassword ? "text" : "password"}
                    placeholder="Re-enter new password"
                    value={resetConfirmPassword}
                    onChange={(e) => setResetConfirmPassword(e.target.value)}
                    minLength={6}
                    required
                  />
                  <button
                    type="button"
                    className="btn-toggle-eye"
                    onClick={() => setShowResetPassword(!showResetPassword)}
                  >
                    {showResetPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              <button type="submit" className="btn-submit-login" disabled={loading}>
                {loading ? (
                  <span className="btn-loading-flex">
                    <RefreshCw size={18} className="spin-icon" />
                    Updating Password...
                  </span>
                ) : (
                  "Update Password & Sign In →"
                )}
              </button>
            </form>

            {message && (
              <div className="form-alert-message" role="alert">
                <Info size={16} />
                <span>{message}</span>
              </div>
            )}

            <div className="form-card-footer text-center">
              <button
                type="button"
                className="btn-subtle-switch"
                onClick={() => {
                  setMessage("");
                  setPage("role_select");
                }}
              >
                &larr; Back to Workspace Portals
              </button>
            </div>
          </div>
        </div>
      </main>
    );
  }

  // ---------------- PATIENT DETAILS PAGE ----------------

  if (page === "patient") {
    return (
      <main className="login-page">
        <section className="login-card patient-card">
          <div className="logo">D</div>
          <h1>{patientEditing ? "Update Medical Profile" : "Complete Patient Profile"}</h1>
          <p className="subtitle">
            Update your age, gender, and medical information.
          </p>

          <form onSubmit={handlePatientSubmit}>
            <label>Full Name</label>
            <input type="text" value={user?.name || patientName} readOnly />

            <label>Email Address</label>
            <input type="email" value={user?.email || email} readOnly />

            <label>Age</label>
            <input
              type="number"
              placeholder="Enter age"
              value={patientAge}
              onChange={(e) => setPatientAge(e.target.value)}
              min="1"
              max="120"
              required
            />

            <label>Gender</label>
            <select
              value={patientGender}
              onChange={(e) => setPatientGender(e.target.value)}
            >
              <option value="">Select gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
              <option value="Prefer not to say">Prefer not to say</option>
            </select>

            <label>Medical History</label>
            <textarea
              rows="3"
              value={medicalHistory}
              onChange={(e) => setMedicalHistory(e.target.value)}
              placeholder="Enter relevant medical history"
            />

            <label>Previous Skin Treatments</label>
            <textarea
              rows="3"
              value={previousSkinTreatments}
              onChange={(e) => setPreviousSkinTreatments(e.target.value)}
              placeholder="Describe previous skin treatments"
            />

            <label>Current Medications</label>
            <textarea
              rows="3"
              value={currentMedications}
              onChange={(e) => setCurrentMedications(e.target.value)}
              placeholder="List current medications"
            />

            <button type="submit" disabled={loading}>
              {loading ? "Saving..." : "Save Patient Profile"}
            </button>
          </form>

          {message && <p className="upload-note" role="alert">{message}</p>}

          {patientEditing && (
            <button
              type="button"
              className="text-button"
              onClick={() => {
                setPatientEditing(false);
                setPatientReturnToProfile(false);
                setPage("dashboard");
                setActivePage("Profile");
                setMessage("");
              }}
            >
              Cancel
            </button>
          )}
        </section>
      </main>
    );
  }

  // ---------------- MAIN DASHBOARD ----------------

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <img src="/dermai-icon.svg" alt="DermAI Logo" className="brand-logo-img" />
          <div className="brand-text-col">
            <span className="brand-title">DermAI</span>
            <span className="brand-sub">Clinical Intelligence</span>
          </div>
        </div>

        <p className="nav-heading">
          {user?.role?.toUpperCase()} WORKSPACE
        </p>

        {navigation.map((item) => (
          <button
            key={item}
            className={`nav-item ${
              activePage === item ? "active" : ""
            }`}
            onClick={() => {
              setActivePage(item);
              setMessage("");
              if (item === "Clinician Approvals") loadPendingClinicians();
              // DATA_PAGES useEffect above handles auto-loading on activePage change
            }}
          >
            {item}
          </button>
        ))}

        <div className="sidebar-bottom">
          <p>{user?.name}</p>
          <p>{user?.role}</p>

          <button
            className="logout-button"
            onClick={handleSignOut}
          >
            Sign Out
          </button>
        </div>
      </aside>

      <main className="dashboard">
        <header className="topbar">
          <div>
            <p className="eyebrow">DERMAI CLINICAL WORKSPACE • {user?.role?.toUpperCase()}</p>
            <h2>{activePage}</h2>
          </div>

          <div className="topbar-right-controls">
            <div className="system-live-pill" title="Backend API and Multimodal Model are active">
              <span className="live-dot"></span>
              <span>AI Online</span>
            </div>

            {/* Section 17 Notification Bell */}
            <div className="notif-wrapper">
              <button
                type="button"
                className="notif-bell-btn"
                onClick={() => setNotifTrayOpen(!notifTrayOpen)}
                aria-label="View notifications"
                title="Notifications & Alerts"
              >
                🔔
                {notifications.filter((n) => !n.read).length > 0 && (
                  <span className="notif-counter-badge">
                    {notifications.filter((n) => !n.read).length}
                  </span>
                )}
              </button>

              {notifTrayOpen && (
                <div className="notif-dropdown-tray">
                  <div className="notif-tray-header">
                    <strong>Alerts & Notifications</strong>
                    <div style={{ display: "flex", gap: "8px" }}>
                      <button
                        type="button"
                        className="notif-header-link"
                        onClick={markAllNotificationsRead}
                      >
                        Mark all read
                      </button>
                      <button
                        type="button"
                        className="notif-header-link"
                        onClick={() => setNotifTrayOpen(false)}
                      >
                        ✕
                      </button>
                    </div>
                  </div>

                  <div className="notif-tray-list">
                    {notifications.length === 0 ? (
                      <p className="notif-empty">No active notifications.</p>
                    ) : (
                      notifications.slice(0, 5).map((n) => (
                        <div
                          key={n.id}
                          className={`notif-tray-item ${n.read ? "read" : "unread"}`}
                          onClick={() => {
                            setNotifications((prev) =>
                              prev.map((item) => (item.id === n.id ? { ...item, read: true } : item))
                            );
                            setActivePage("Notifications & Alerts");
                            setNotifTrayOpen(false);
                          }}
                        >
                          <div className="notif-item-top">
                            <span className={`notif-badge-pill pill-${n.level.toLowerCase()}`}>
                              {n.level}
                            </span>
                            <span className="notif-time">{n.timestamp}</span>
                          </div>
                          <strong className="notif-title">{n.title}</strong>
                          <p className="notif-msg">{n.message}</p>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="notif-tray-footer">
                    <button
                      type="button"
                      onClick={() => {
                        setActivePage("Notifications & Alerts");
                        setNotifTrayOpen(false);
                      }}
                    >
                      Open Full Alerts Center →
                    </button>
                  </div>
                </div>
              )}
            </div>

            <button
              type="button"
              className="user-avatar"
              onClick={() => { setActivePage("Profile"); setMessage(""); }}
              aria-label="Open profile"
              title="Open Profile"
            >
              {user?.name?.charAt(0).toUpperCase() || "D"}
            </button>
          </div>
        </header>

        {/* USER DASHBOARD */}

        {activePage === "Dashboard" && user?.role === "user" && (
          <div className="modern-dashboard-layout">
            {/* 1. Hero Clinical Banner */}
            <section className="dashboard-hero-banner">
              <div className="hero-content">
                <div className="hero-status-pill">
                  <span className="live-pulse"></span>
                  <span>AI Neural Engine Active • EfficientNet-B0</span>
                </div>
                <h1>Good day, {user.name} 👋</h1>
                <p className="hero-description">
                  Welcome to your clinical skin disease pre-screening workspace. Capture or upload mobile lesion photographs to run multimodal analysis, disease risk triage, and Grad-CAM visual explainability.
                </p>
                <div className="hero-actions">
                  <button
                    className="btn-hero-primary"
                    onClick={() => setActivePage("New Analysis")}
                  >
                    ⚡ Start New Lesion Analysis
                  </button>
                  <button
                    className="btn-hero-secondary"
                    onClick={() => setActivePage("Reports")}
                  >
                    📑 View Diagnostic Reports
                  </button>
                  <button
                    className="btn-hero-tertiary"
                    onClick={() => setActivePage("Profile")}
                  >
                    👤 Profile: Age {patientProfile?.age ?? user?.age ?? 55} ({patientProfile?.gender ?? "Not specified"})
                  </button>
                </div>
              </div>

              <div className="hero-metrics-pill">
                <div className="metric-row">
                  <span className="metric-label">Model Architecture</span>
                  <strong className="metric-val">Multimodal CNN + 64-dim MLP</strong>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Diagnostic Benchmark</span>
                  <strong className="metric-val">PAD-UFES-20 (2,298 Cases)</strong>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Test Accuracy</span>
                  <strong className="metric-val highlight-green">76.74% (Held-Out)</strong>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Explainability</span>
                  <strong className="metric-val">Grad-CAM Class Activation</strong>
                </div>
              </div>
            </section>

            {/* 2. Key Metrics Grid */}
            <div className="section-header-row">
              <h3>Clinical Overview & History</h3>
              <span className="section-subtitle">Real-time statistics from your recorded screenings</span>
            </div>

            <section className="stats-grid-enhanced">
              <article className="enhanced-stat-card">
                <div className="stat-icon-wrap icon-blue">🔬</div>
                <div>
                  <span className="stat-label">Total Screenings</span>
                  <h2 className="stat-number">{dashboardData?.total_cases ?? 0}</h2>
                  <p className="stat-sub">Recorded mobile cases</p>
                </div>
              </article>

              <article className="enhanced-stat-card">
                <div className="stat-icon-wrap icon-green">🎯</div>
                <div>
                  <span className="stat-label">AI Predictions Run</span>
                  <h2 className="stat-number">{dashboardData?.total_predictions ?? 0}</h2>
                  <p className="stat-sub">Multimodal evaluations</p>
                </div>
              </article>

              <article className="enhanced-stat-card">
                <div className="stat-icon-wrap icon-amber">🕒</div>
                <div>
                  <span className="stat-label">Latest Activity</span>
                  <h2 className="stat-number stat-date">{dashboardData?.recent_activity ?? "No scans"}</h2>
                  <p className="stat-sub">{dashboardData?.recent_activity ? "Last screening timestamp" : "Awaiting first case"}</p>
                </div>
              </article>

              <article className="enhanced-stat-card">
                <div className="stat-icon-wrap icon-teal">🧬</div>
                <div>
                  <span className="stat-label">Patient Calibration</span>
                  <h2 className="stat-number stat-age">
                    Age {patientProfile?.age ?? user?.age ?? 55}
                  </h2>
                  <p className="stat-sub">
                    <button type="button" className="text-link-btn" onClick={() => setActivePage("Profile")}>
                      Update Clinical Profile →
                    </button>
                  </p>
                </div>
              </article>
            </section>

            {/* 3. Recent Screenings Table */}
            <section className="recent-cases-section">
              <div className="recent-cases-header">
                <div>
                  <h4>Recent Analysis History</h4>
                  <p>Detailed breakdown of your latest lesion submissions</p>
                </div>
                {(dashboardData?.cases || []).length > 0 && (
                  <button
                    className="btn-view-all"
                    onClick={() => setActivePage("Reports")}
                  >
                    View All & Export →
                  </button>
                )}
              </div>

              {(dashboardData?.cases || []).length === 0 ? (
                <div className="empty-cases-card">
                  <div className="empty-icon">📷</div>
                  <h5>No skin analyses recorded yet</h5>
                  <p>Upload your first clear mobile photograph of a skin lesion to receive an instant multimodal prediction and explainability heatmap.</p>
                  <button
                    className="btn-start-first"
                    onClick={() => setActivePage("New Analysis")}
                  >
                    + Submit Your First Image
                  </button>
                </div>
              ) : (
                <div className="dashboard-cases-table-wrap">
                  <table className="dashboard-cases-table">
                    <thead>
                      <tr>
                        <th>Case</th>
                        <th>Date & Time</th>
                        <th>Affected Region</th>
                        <th>Predicted Finding</th>
                        <th>Confidence</th>
                        <th>Risk Stratification</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(dashboardData.cases.slice(0, 5)).map((c) => (
                        <tr key={c.case_id}>
                          <td>
                            <div className="case-id-cell">
                              {c.image_url ? (
                                <img src={c.image_url} alt="" className="table-thumb" />
                              ) : (
                                <span className="table-thumb-placeholder">🔬</span>
                              )}
                              <strong>#{c.case_id}</strong>
                            </div>
                          </td>
                          <td>{c.created_at || "—"}</td>
                          <td>
                            <span className="body-part-pill">{c.affected_body_part || "Not specified"}</span>
                          </td>
                          <td>
                            <strong>{c.disease_name || c.predicted_class || "Processing"}</strong>
                          </td>
                          <td>
                            {c.confidence != null ? (
                              <div className="table-conf-bar">
                                <div className="bar-track">
                                  <div
                                    className="bar-fill"
                                    style={{ width: `${Math.min(c.confidence, 100)}%` }}
                                  ></div>
                                </div>
                                <span>{c.confidence}%</span>
                              </div>
                            ) : "—"}
                          </td>
                          <td>
                            {c.risk_level && (
                              <span className={`risk-badge risk-${c.risk_level.toLowerCase().replace(/\s+/g, "-")}`}>
                                {c.risk_level}
                              </span>
                            )}
                          </td>
                          <td>
                            <div style={{ display: "flex", gap: "6px" }}>
                              <button
                                type="button"
                                className="btn-table-action"
                                onClick={() => {
                                  exportCasePDF(c, c.image_url);
                                }}
                                title="Download PDF Report"
                              >
                                📄 PDF
                              </button>
                              <button
                                type="button"
                                className="btn-table-action"
                                style={{ background: "#ecfdf5", color: "#065f46", borderColor: "#a7f3d0" }}
                                onClick={() => handleResendEmailReport(c.case_id)}
                                title="Email Diagnostic Alert to Patient"
                              >
                                ✉️ Email
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </div>
        )}

        {/* CLINICIAN DASHBOARD */}

        {activePage === "Dashboard" &&
          user?.role === "clinician" && (
            <>
              <section className="welcome-banner">
                <div>
                  <p className="eyebrow">CLINICIAN WORKSPACE</p>
                  <h1>Welcome, {user.name}</h1>

                  <p>
                    Review patient cases and monitor prediction
                    results.
                  </p>
                </div>

                <div className="banner-symbol">D</div>
              </section>

              <h3>Clinical Overview</h3>

              <section className="stats-grid">
                <article
                  className="stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => {
                    setClinicianCaseFilter("all");
                    setActivePage("Total Cases");
                  }}
                >
                  <span>Total Cases</span>
                  <h2>{dashboardData?.total_cases ?? 0}</h2>
                  <p>Recorded clinical cases →</p>
                </article>

                <article
                  className="stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => {
                    setClinicianCaseFilter("pending");
                    setActivePage("Total Cases");
                  }}
                >
                  <span>Cases Requiring Review</span>
                  <h2 style={{ color: (dashboardData?.pending_reviews || 0) > 0 ? "#d97706" : "inherit" }}>
                    {dashboardData?.pending_reviews ?? 0}
                  </h2>
                  <p>Pending clinician review →</p>
                </article>

                <article
                  className="stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => {
                    setActivePage("Total Cases");
                  }}
                >
                  <span>High Risk Cases</span>
                  <h2 style={{ color: (dashboardData?.high_risk_cases || 0) > 0 ? "#dc2626" : "inherit" }}>
                    {dashboardData?.high_risk_cases ?? 0}
                  </h2>
                  <p>Require urgent attention →</p>
                </article>
              </section>
            </>
          )}

        {/* ADMIN DASHBOARD */}

        {activePage === "Dashboard" &&
          user?.role === "admin" && (
            <div className="user-dashboard-wrapper">
              {/* 1. Executive Admin Hero Banner */}
              <section className="admin-hero-banner">
                <div className="admin-hero-top">
                  <div className="admin-badges-wrap">
                    <span className="admin-role-pill">
                      🛡️ Administrator Platform Hub
                    </span>
                    <span className="admin-status-pill">
                      <span className="admin-pulse-dot" />
                      Production Cluster Online · Multimodal Model Active
                    </span>
                  </div>

                  <button
                    type="button"
                    className="admin-hero-refresh-btn"
                    onClick={loadDashboard}
                    disabled={dashboardLoading}
                  >
                    {dashboardLoading ? "Refreshing…" : "🔄 Refresh Data"}
                  </button>
                </div>

                <div className="admin-hero-body">
                  <div className="admin-hero-text">
                    <h1>Welcome, {user.name}</h1>
                    <p>
                      Supervise platform diagnostic volume, audit registered clinical accounts, inspect multimodal neural performance, and manage clinician approvals.
                    </p>
                  </div>

                  <div className="admin-hero-actions">
                    <button
                      type="button"
                      className="btn-hero-primary"
                      onClick={() => setActivePage("ML Analytics")}
                    >
                      📊 Model Analytics Suite →
                    </button>
                    <button
                      type="button"
                      className="btn-hero-secondary"
                      onClick={() => setActivePage("Clinician Approvals")}
                    >
                      🩺 Review Clinicians ({dashboardData?.pending_clinicians_count ?? 0})
                    </button>
                  </div>
                </div>

                <div className="admin-stats-grid">
                  <div className="admin-stat-cell">
                    <div className="admin-stat-label">
                      <span>1. Train Accuracy</span>
                      <span style={{ color: "#34d399", fontSize: "0.85rem" }}>★ Peak</span>
                    </div>
                    <div className="admin-stat-value" style={{ color: "#34d399" }}>98.07%</div>
                    <div className="admin-stat-sub">Multimodal Convergence</div>
                  </div>

                  <div className="admin-stat-cell">
                    <div className="admin-stat-label">
                      <span>2. Val Accuracy</span>
                      <span style={{ color: "#a7f3d0", fontSize: "0.85rem" }}>Optimal</span>
                    </div>
                    <div className="admin-stat-value">77.55%</div>
                    <div className="admin-stat-sub">Cross-Validation Peak</div>
                  </div>

                  <div className="admin-stat-cell">
                    <div className="admin-stat-label">
                      <span>3. Test Accuracy</span>
                      <span style={{ color: "#a7f3d0", fontSize: "0.85rem" }}>Holdout</span>
                    </div>
                    <div className="admin-stat-value">76.74%</div>
                    <div className="admin-stat-sub">PAD-UFES-20 Benchmark</div>
                  </div>

                  <div className="admin-stat-cell">
                    <div className="admin-stat-label">
                      <span>4. Clinical Scope</span>
                      <span style={{ color: "#a7f3d0", fontSize: "0.85rem" }}>6 Classes</span>
                    </div>
                    <div className="admin-stat-value">2,298</div>
                    <div className="admin-stat-sub">EfficientNet-B0 Backbone</div>
                  </div>
                </div>
              </section>

              {/* 2. Platform KPI Metrics Grid */}
              <div className="section-header-row">
                <h3>Core Platform Metrics</h3>
                <span className="section-subtitle">Real-time telemetry and user statistics</span>
              </div>

              <section className="stats-grid-enhanced">
                <article
                  className="enhanced-stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => setActivePage("Registered Users")}
                >
                  <div className="stat-icon-wrap icon-blue">👥</div>
                  <div>
                    <span className="stat-label">Total Accounts</span>
                    <h2 className="stat-number">{dashboardData?.total_users ?? 0}</h2>
                    <p className="stat-sub">Manage registered users →</p>
                  </div>
                </article>

                <article
                  className="enhanced-stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => setActivePage("Total Analyses")}
                >
                  <div className="stat-icon-wrap icon-emerald">⚡</div>
                  <div>
                    <span className="stat-label">Total Analyses</span>
                    <h2 className="stat-number">{dashboardData?.total_analyses ?? 0}</h2>
                    <p className="stat-sub">AI screenings executed →</p>
                  </div>
                </article>

                <article
                  className="enhanced-stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => setActivePage("Clinician Approvals")}
                >
                  <div className="stat-icon-wrap icon-amber">🩺</div>
                  <div>
                    <span className="stat-label">Pending Verification</span>
                    <h2 className="stat-number" style={{ color: (dashboardData?.pending_clinicians_count || 0) > 0 ? "#d97706" : "inherit" }}>
                      {dashboardData?.pending_clinicians_count ?? 0}
                    </h2>
                    <p className="stat-sub">Clinician approvals →</p>
                  </div>
                </article>

                <article
                  className="enhanced-stat-card"
                  style={{ cursor: "pointer" }}
                  onClick={() => setActivePage("Platform Health")}
                >
                  <div className="stat-icon-wrap icon-teal">🛡️</div>
                  <div>
                    <span className="stat-label">Platform Status</span>
                    <h2 className="stat-number" style={{ fontSize: "1.35rem", color: "#0d9488" }}>
                      {dashboardData?.model_info?.status === "loaded" ? "✓ Online" : "—"}
                    </h2>
                    <p className="stat-sub">EfficientNet-B0 Model →</p>
                  </div>
                </article>
              </section>

              {/* 3. Administrative Quick Navigation Grid */}
              <div className="section-header-row">
                <h3>Administrative Controls</h3>
                <span className="section-subtitle">Direct shortcuts to system modules</span>
              </div>

              <section className="admin-actions-grid">
                <div
                  className="admin-action-card"
                  onClick={() => setActivePage("Registered Users")}
                >
                  <div>
                    <div className="admin-card-icon icon-blue">👥</div>
                    <h5>User Directory</h5>
                    <p>Audit user accounts, assign roles (Admin / Clinician / User), and oversee platform security permissions.</p>
                  </div>
                  <span className="admin-card-cta">Manage Users →</span>
                </div>

                <div
                  className="admin-action-card"
                  onClick={() => setActivePage("Clinician Approvals")}
                >
                  <div>
                    <div className="admin-card-icon icon-amber">🩺</div>
                    <h5>Clinician Verifications</h5>
                    <p>Review medical license credentials, verify dermatology practitioners, and grant clinical access.</p>
                  </div>
                  <span className="admin-card-cta">
                    Review Pending ({(dashboardData?.pending_clinicians_count ?? 0)}) →
                  </span>
                </div>

                <div
                  className="admin-action-card"
                  onClick={() => setActivePage("ML Analytics")}
                >
                  <div>
                    <div className="admin-card-icon icon-emerald">📊</div>
                    <h5>ML Analytics & Benchmark</h5>
                    <p>Examine the 4-architecture evaluation matrix, ROC curves, metadata fusion layers, and Grad-CAM maps.</p>
                  </div>
                  <span className="admin-card-cta">Explore Benchmark →</span>
                </div>

                <div
                  className="admin-action-card"
                  onClick={() => setActivePage("Total Analyses")}
                >
                  <div>
                    <div className="admin-card-icon icon-purple">📈</div>
                    <h5>Diagnostic Volume</h5>
                    <p>Track real-time disease distribution, clinical frequency breakdown across 6 lesion types, and telemetry.</p>
                  </div>
                  <span className="admin-card-cta">View Analytics →</span>
                </div>
              </section>

              {/* 4. Recent Accounts & Disease Distribution Split View */}
              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "20px", marginTop: "10px" }}>
                {/* Recent Registered Users */}
                <section className="recent-cases-section">
                  <div className="recent-cases-header">
                    <div>
                      <h4 style={{ margin: "0 0 4px 0", fontSize: "1.1rem", color: "#0f172a" }}>Recent Registered Accounts</h4>
                      <p style={{ margin: 0, fontSize: "0.85rem", color: "#64748b" }}>Latest users onboarded to the DermAI clinical network</p>
                    </div>
                    <button
                      type="button"
                      className="btn-view-all"
                      onClick={() => setActivePage("Registered Users")}
                    >
                      View All →
                    </button>
                  </div>

                  {(dashboardData?.users || []).length === 0 ? (
                    <p style={{ color: "#94a3b8", textAlign: "center", padding: "20px" }}>No registered users found.</p>
                  ) : (
                    <div className="dashboard-cases-table-wrap">
                      <table className="dashboard-cases-table">
                        <thead>
                          <tr>
                            <th>User</th>
                            <th>Role</th>
                            <th>Status</th>
                            <th>Joined</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(dashboardData.users.slice(0, 5)).map((u) => (
                            <tr key={u.user_id}>
                              <td>
                                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                                  <div style={{
                                    width: "32px",
                                    height: "32px",
                                    borderRadius: "50%",
                                    background: "#e0f2fe",
                                    color: "#0369a1",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    fontWeight: 700,
                                    fontSize: "0.85rem"
                                  }}>
                                    {u.name?.charAt(0).toUpperCase() || "U"}
                                  </div>
                                  <div>
                                    <strong style={{ display: "block", fontSize: "0.88rem", color: "#0f172a" }}>{u.name}</strong>
                                    <small style={{ color: "#64748b", fontSize: "0.78rem" }}>{u.email}</small>
                                  </div>
                                </div>
                              </td>
                              <td>
                                <span className={`risk-badge risk-${u.role === "admin" ? "high-risk" : u.role === "clinician" ? "moderate-risk" : "low-risk"}`}>
                                  {u.role}
                                </span>
                              </td>
                              <td>
                                <span style={{
                                  fontSize: "0.76rem",
                                  fontWeight: 600,
                                  color: u.is_approved ? "#059669" : "#d97706",
                                  background: u.is_approved ? "#ecfdf5" : "#fffbeb",
                                  padding: "3px 8px",
                                  borderRadius: "6px"
                                }}>
                                  {u.is_approved ? "✓ Approved" : "⏳ Pending"}
                                </span>
                              </td>
                              <td style={{ fontSize: "0.82rem", color: "#64748b" }}>
                                {u.created_at ? u.created_at.split("T")[0] : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>

                {/* Disease Distribution Preview */}
                <section className="recent-cases-section">
                  <div className="recent-cases-header">
                    <div>
                      <h4 style={{ margin: "0 0 4px 0", fontSize: "1.1rem", color: "#0f172a" }}>Screening Distribution</h4>
                      <p style={{ margin: 0, fontSize: "0.85rem", color: "#64748b" }}>Clinical cases diagnosed by AI across classes</p>
                    </div>
                    <button
                      type="button"
                      className="btn-view-all"
                      onClick={() => setActivePage("Total Analyses")}
                    >
                      Details →
                    </button>
                  </div>

                  {Object.keys(dashboardData?.disease_distribution || {}).length === 0 ? (
                    <p style={{ color: "#94a3b8", textAlign: "center", padding: "20px" }}>No predictions recorded yet.</p>
                  ) : (
                    <div className="prob-bars-list" style={{ marginTop: "10px" }}>
                      {Object.entries(dashboardData.disease_distribution)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 6)
                        .map(([cls, count]) => {
                          const total = Object.values(dashboardData.disease_distribution).reduce(
                            (s, v) => s + v, 0
                          );
                          const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                          return (
                            <div key={cls} className="prob-bar-row">
                              <div className="prob-labels">
                                <span style={{ fontSize: "0.82rem" }}>
                                  <strong style={{ color: "#0f766e" }}>{cls}</strong> — {DISEASE_NAMES_MAP[cls] || cls}
                                </span>
                                <span style={{ fontWeight: 700, fontSize: "0.82rem" }}>{count} ({pct}%)</span>
                              </div>
                              <div className="prob-bar-track">
                                <div className="prob-bar-fill" style={{ width: `${pct}%`, background: "linear-gradient(90deg, #0d9488, #10b981)" }} />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </section>
              </div>
            </div>
          )}

        {/* NEW IMAGE ANALYSIS */}

        {activePage === "New Analysis" &&
          user?.role === "user" && (
            <section className="info-panel upload-panel">
              <h3>New Skin Analysis</h3>
              <p>
                Tell us about the skin condition, then select or capture
                a clear image. These details will be saved with this case.
              </p>

              <h4>1. Skin Condition Details</h4>

              <label htmlFor="affected-body-part">Affected Body Part</label>
              <select
                id="affected-body-part"
                value={affectedBodyPart}
                onChange={(e) => setAffectedBodyPart(e.target.value)}
                disabled={isUploading}
                required
              >
                <option value="">Select body part</option>
                <option value="Face">Face</option>
                <option value="Scalp">Scalp</option>
                <option value="Neck">Neck</option>
                <option value="Chest">Chest</option>
                <option value="Back">Back</option>
                <option value="Arm">Arm</option>
                <option value="Hand">Hand</option>
                <option value="Leg">Leg</option>
                <option value="Foot">Foot</option>
                <option value="Other">Other</option>
              </select>

              <label htmlFor="condition-duration">
                How long have you had this condition?
              </label>
              <select
                id="condition-duration"
                value={conditionDuration}
                onChange={(e) => setConditionDuration(e.target.value)}
                disabled={isUploading}
                required
              >
                <option value="">Select duration</option>
                <option value="Less than 1 week">Less than 1 week</option>
                <option value="1–4 weeks">1–4 weeks</option>
                <option value="1–3 months">1–3 months</option>
                <option value="3–6 months">3–6 months</option>
                <option value="More than 6 months">More than 6 months</option>
                <option value="Not sure">Not sure</option>
              </select>

              <label>What symptoms are you experiencing?</label>
              <div className="symptoms-list">
                {[
                  "Itching",
                  "Pain",
                  "Bleeding",
                  "Change in size",
                  "Change in color",
                  "Swelling",
                  "None of these",
                ].map((symptom) => (
                  <label key={symptom} className="symptom-option">
                    <input
                      type="checkbox"
                      checked={symptoms.includes(symptom)}
                      disabled={isUploading}
                      onChange={(e) => {
                        if (symptom === "None of these") {
                          setSymptoms(e.target.checked ? [symptom] : []);
                        } else {
                          setSymptoms((current) => {
                            const withoutNone = current.filter(
                              (item) => item !== "None of these"
                            );
                            return e.target.checked
                              ? [...withoutNone, symptom]
                              : withoutNone.filter((item) => item !== symptom);
                          });
                        }
                      }}
                    />
                    <span>{symptom}</span>
                  </label>
                ))}
              </div>

              <h4>2. Skin Image</h4>
              <p>Choose an image from your device or capture one using your camera.</p>

              <div className="image-source-options">
                <label className="upload-box" htmlFor="skin-image">
                  <span className="upload-icon">↑</span>
                  <strong>Choose from Device</strong>
                  <span>JPG or PNG, maximum 10 MB</span>
                  <input
                    id="skin-image"
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={handleImageSelect}
                    disabled={isUploading}
                  />
                </label>

                <button type="button" className="upload-box camera-upload-box"
                  onClick={startCamera} disabled={isUploading}>
                  <span className="upload-icon">📷</span>
                  <strong>Capture Using Camera</strong>
                  <span>Open your camera and take a photo</span>
                </button>
              </div>

              {cameraOpen && (
                <div className="camera-modal-backdrop">
                  <section className="camera-modal" role="dialog" aria-modal="true">
                    <div className="camera-modal-header">
                      <h3>Capture Skin Image</h3>
                      <button type="button" className="camera-close-button" onClick={stopCamera}>Close</button>
                    </div>
                    {cameraError ? (
                      <p className="camera-error" role="alert">{cameraError}</p>
                    ) : (
                      <video ref={videoRef} className="camera-preview" autoPlay muted playsInline />
                    )}
                    <div className="camera-actions">
                      <button type="button" className="camera-secondary-button" onClick={stopCamera}>Cancel</button>
                      <button type="button" onClick={capturePhoto} disabled={Boolean(cameraError)}>Capture Photo</button>
                    </div>
                  </section>
                </div>
              )}

              {selectedImage && (
                <div className="image-preview">
                  <img
                    src={selectedImage.preview}
                    alt="Selected skin image"
                  />
                  <p>{selectedImage.file.name}</p>
                  <button
                    type="button"
                    onClick={removeImage}
                    disabled={isUploading}
                  >
                    Remove Image
                  </button>
                </div>
              )}

              {selectedImage && (
                <section className="quality-assessment-panel" aria-live="polite">
                  <h4>Image Quality & Preprocessing Status</h4>
                  {isAssessingQuality ? (
                    <div className="assessment-loading-box">
                      <p className="upload-note">
                        ⏳ <strong>Assessing image quality...</strong><br />
                        Validating resolution, focus sharpness, brightness, and contrast.
                      </p>
                    </div>
                  ) : qualityAssessment ? (
                    <>
                      <div className="quality-status-badge-row">
                        <span className={`status-pill ${qualityAssessment.is_acceptable ? "pill-success" : "pill-warning"}`}>
                          {qualityAssessment.is_acceptable ? "✓ Image Quality Check Passed" : "⚠️ Quality Warning"}
                        </span>
                        <span className="pipeline-pill">Automated Preprocessing Ready</span>
                      </div>

                      <div className="quality-metrics">
                        <p><strong>Resolution:</strong> {qualityAssessment.width && qualityAssessment.height ? `${qualityAssessment.width} × ${qualityAssessment.height} px` : "Verified"}</p>
                        <p><strong>Sharpness:</strong> {qualityAssessment.sharpness ?? "Good"}</p>
                        <p><strong>Brightness:</strong> {qualityAssessment.brightness ?? "Optimal"}</p>
                        <p><strong>Contrast:</strong> {qualityAssessment.contrast ?? "Good"}</p>
                      </div>

                      {/* Dynamic Preprocessing Step Notification */}
                      <div className="preprocessing-step-banner">
                        <span className="step-icon">🔬</span>
                        <div className="step-info">
                          <strong>Clinical Preprocessing Pipeline:</strong>
                          <p>
                            The uploaded image is verified, automatically cropped & resized to <strong>224 × 224 pixels</strong>,
                            normalized with ImageNet statistics, and processed for multimodal classifier inference.
                          </p>
                        </div>
                      </div>

                      {qualityAssessment.issues?.length > 0 && (
                        <div className="quality-issues-alert">
                          <strong>Quality Notes:</strong>
                          <ul>{qualityAssessment.issues.map((item, index) => <li key={`issue-${index}`}>{item}</li>)}</ul>
                        </div>
                      )}
                    </>
                  ) : null}
                </section>
              )}

              <button
                type="button"
                onClick={uploadImage}
                disabled={!selectedImage || isUploading || isAssessingQuality || !qualityAssessment?.is_acceptable}
              >
                {isUploading ? "Submitting..." : "Submit for Analysis"}
              </button>

              {uploadStatus && (
                <p className="upload-note" role="status">
                  {typeof uploadStatus === "string"
                    ? uploadStatus
                    : getReadableError(uploadStatus, "Please try again.")}
                </p>
              )}

              {analysisResult && (
                <section className="quality-assessment-panel analysis-result-panel" aria-live="polite">
                  <div className="analysis-header-row">
                    <div>
                      <span className="analysis-subheading">AI Dermatological Analysis</span>
                      <h4 className="analysis-condition-title">
                        {analysisResult.disease_name || analysisResult.predicted_class}
                        <span className="disease-code-pill">({analysisResult.predicted_class})</span>
                      </h4>
                      {analysisResult.disease_category && (
                        <p className="disease-category-text">{analysisResult.disease_category}</p>
                      )}
                    </div>
                    {analysisResult.risk_level && (
                      <div className={`risk-badge risk-${analysisResult.risk_level.toLowerCase().replace(/\s+/g, "-")}`}>
                        {analysisResult.risk_level === "High Risk" && "⚠️ "}
                        {analysisResult.risk_level === "Moderate Risk" && "⚡ "}
                        {analysisResult.risk_level === "Low Risk" && "✓ "}
                        {analysisResult.risk_level === "Requires Clinical Review" && "ℹ️ "}
                        {analysisResult.risk_level}
                      </div>
                    )}
                  </div>

                  {analysisResult.disease_description && (
                    <p className="disease-description-box">{analysisResult.disease_description}</p>
                  )}

                  {/* Explainable AI: Grad-CAM Attention Heatmap */}
                  {analysisResult.gradcam_image && (
                    <div className="explainability-comparison-container">
                      <h5>Explainable AI — Grad-CAM Visual Heatmap & Attention Analysis</h5>
                      <p className="xai-caption">
                        Grad-CAM visualizes the exact morphological regions of the skin lesion that triggered the AI model's diagnosis.
                        Warm colors (red, yellow) show peak neural activation, while cool colors (blue) represent non-influential background tissue.
                      </p>
                      <div className="xai-image-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
                        {lastAnalyzedImage && (
                          <div className="xai-card">
                            <span className="xai-label">1. Original Lesion</span>
                            <img src={lastAnalyzedImage} alt="Original uploaded lesion" className="xai-img" />
                          </div>
                        )}
                        {analysisResult.gradcam_raw_heatmap && (
                          <div className="xai-card">
                            <span className="xai-label">2. Pure Activation Heatmap</span>
                            <img src={analysisResult.gradcam_raw_heatmap} alt="Grad-CAM thermal heatmap" className="xai-img" />
                          </div>
                        )}
                        <div className="xai-card">
                          <span className="xai-label">{analysisResult.gradcam_raw_heatmap ? "3. AI Attention Overlay" : "AI Attention Overlay"}</span>
                          <img src={analysisResult.gradcam_image} alt="Grad-CAM heatmap overlay" className="xai-img" />
                        </div>
                      </div>
                      <div className="xai-heatmap-legend">
                        <span>Low Activation (Normal Skin)</span>
                        <div className="heat-gradient-bar"></div>
                        <span>Peak Activation (Lesion Trigger)</span>
                      </div>
                    </div>
                  )}

                  {/* Clinical Risk & Treatment Categorization Module */}
                  <div className="treatment-decision-panel">
                    <div className="treatment-header-row">
                      <div>
                        <span className="treatment-eyebrow">RISK STRATIFICATION & TREATMENT MODULE</span>
                        <h4 className="treatment-title">Does this lesion require medical treatment?</h4>
                      </div>
                      <div className={`treatment-status-badge ${
                        (analysisResult.risk_level || "").toLowerCase().includes("high")
                          ? "treatment-badge-urgent"
                          : (analysisResult.risk_level || "").toLowerCase().includes("moderate")
                          ? "treatment-badge-moderate"
                          : "treatment-badge-low"
                      }`}>
                        {(analysisResult.risk_level || "").toLowerCase().includes("high") && "🚨 "}
                        {(analysisResult.risk_level || "").toLowerCase().includes("moderate") && "⚡ "}
                        {(analysisResult.risk_level || "").toLowerCase().includes("low") && "✓ "}
                        {analysisResult.treatment_required || (
                          (analysisResult.risk_level || "").toLowerCase().includes("high")
                            ? "Yes — Prompt Treatment Required"
                            : (analysisResult.risk_level || "").toLowerCase().includes("moderate")
                            ? "Yes — Treatment Advised"
                            : "No Treatment Required"
                        )}
                      </div>
                    </div>

                    <div className="treatment-grid">
                      <div className="treatment-card">
                        <span className="treatment-label">Condition Severity</span>
                        <strong style={{
                          color: (analysisResult.risk_level || "").toLowerCase().includes("high")
                            ? "#c0392b"
                            : (analysisResult.risk_level || "").toLowerCase().includes("moderate")
                            ? "#d35400"
                            : "#27ae60"
                        }}>
                          {analysisResult.severity || analysisResult.risk_level}
                        </strong>
                      </div>

                      <div className="treatment-card">
                        <span className="treatment-label">Risk Category</span>
                        <strong>{analysisResult.risk_level}</strong>
                      </div>

                      <div className="treatment-card full-width">
                        <span className="treatment-label">Recommended Clinical Procedure</span>
                        <p>{analysisResult.treatment_type || analysisResult.clinical_recommendation}</p>
                      </div>
                    </div>
                  </div>

                  {/* Model Confidence Meter */}
                  <div className="confidence-overview-box">
                    <div className="confidence-header">
                      <span>Model Confidence ({analysisResult.model_name || "EfficientNet-B0 Multimodal"})</span>
                      <strong className="confidence-num">{analysisResult.confidence}%</strong>
                    </div>
                    <div className="confidence-bar-bg">
                      <div
                        className="confidence-bar-fill"
                        style={{ width: `${Math.min(Math.max(analysisResult.confidence, 5), 100)}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Class Probabilities Progress Breakdown */}
                  <div className="quality-metrics probability-breakdown">
                    <strong>Multimodal Class Probabilities</strong>
                    <div className="prob-bars-list">
                      {Object.entries(analysisResult.class_probabilities || {})
                        .sort((a, b) => b[1] - a[1])
                        .map(([label, score]) => (
                          <div key={label} className="prob-bar-row">
                            <div className="prob-labels">
                              <span className="prob-name">
                                <strong>{label}</strong> {DISEASE_NAMES_MAP[label] ? `— ${DISEASE_NAMES_MAP[label]}` : ""}
                              </span>
                              <span className="prob-pct">{score}%</span>
                            </div>
                            <div className="prob-bar-track">
                              <div
                                className={`prob-bar-fill ${label === analysisResult.predicted_class ? "active-prob" : ""}`}
                                style={{ width: `${Math.max(score, 1)}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>

                  {/* Clinical Recommendation Box */}
                  {analysisResult.clinical_recommendation && (
                    <div className="clinical-recommendation-box">
                      <strong>Clinical Next Steps:</strong>
                      <p>{analysisResult.clinical_recommendation}</p>
                    </div>
                  )}

                  {/* Section: Automated Email Report Dispatch Status Banner */}
                  <div className="email-dispatch-status-banner">
                    <div className="email-status-left">
                      <span className="email-status-icon">✉️</span>
                      <div>
                        <strong style={{ display: "block", color: "#065f46", fontSize: "0.95rem" }}>
                          Automated Clinical Report Alert Dispatched
                        </strong>
                        <span style={{ fontSize: "0.85rem", color: "#047857" }}>
                          A preliminary triage alert and diagnostic summary were sent directly to <strong>{emailAlertInfo?.recipientEmail || user?.email}</strong>.
                        </span>
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn-resend-email"
                      onClick={() => handleResendEmailReport(emailAlertInfo?.caseId || analysisResult?.case_id)}
                      disabled={isSendingEmail}
                    >
                      {isSendingEmail ? "Sending..." : "🔄 Resend Email Alert"}
                    </button>
                  </div>

                  {/* One-Click Report Download Action Toolbar */}
                  <div className="report-export-toolbar">
                    <span className="toolbar-label">📥 Export Diagnostic Report:</span>
                    <div className="toolbar-buttons">
                      <button
                        type="button"
                        className="btn-export-pdf"
                        onClick={() => exportCasePDF(analysisResult, analysisResult.gradcam_image)}
                      >
                        📄 Download PDF Report
                      </button>
                      <button
                        type="button"
                        className="btn-export-csv"
                        onClick={() => exportCaseCSV(analysisResult)}
                      >
                        📊 Download CSV
                      </button>
                      <button
                        type="button"
                        className="btn-export-json"
                        onClick={() => exportCaseJSON(analysisResult)}
                      >
                        💾 Download JSON
                      </button>
                      <button
                        type="button"
                        className="btn-export-email"
                        onClick={() => handleResendEmailReport(emailAlertInfo?.caseId || analysisResult?.case_id)}
                        disabled={isSendingEmail}
                      >
                        ✉️ {isSendingEmail ? "Sending..." : "Email Report"}
                      </button>
                    </div>
                  </div>

                  <p className="upload-note disclaimer-note">
                    <strong>Medical Screening Notice:</strong> This preliminary analysis was generated by the DermAI multimodal screening model ({analysisResult.model_name}). It is intended as a clinical decision-support tool, not an autonomous medical diagnosis. Please consult a qualified clinician or dermatologist for definitive assessment.
                  </p>
                </section>
              )}
            </section>
          )}

        {activePage === "Clinician Approvals" && user?.role === "admin" && (
          <section className="info-panel">
            <h3>Clinician Registration Requests</h3>
            <p>Review clinician accounts and approve access to DermAI.</p>
            <button type="button" onClick={loadPendingClinicians}>
              Refresh Requests
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            {pendingClinicians.length === 0 ? (
              <p>No pending clinician requests.</p>
            ) : (
              pendingClinicians.map((clinician) => (
                <article className="stat-card" key={clinician.user_id}>
                  <strong>{clinician.name}</strong>
                  <p>{clinician.email}</p>
                  <button
                    type="button"
                    onClick={() => approveClinician(clinician.user_id)}
                  >
                    Approve Clinician
                  </button>
                </article>
              ))
            )}
          </section>
        )}

        {/* ============================================================
             USER PAGES
             ============================================================ */}

        {/* ---- USER: Previous Predictions (full case list) ---- */}
        {activePage === "Previous Predictions" && user?.role === "user" && (
            <section className="info-panel">
              <h3>Previous Predictions</h3>
              <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
                {dashboardLoading ? "Loading…" : "Refresh"}
              </button>
              {message && <p className="upload-note" role="status">{message}</p>}
              {(dashboardData?.cases || []).length === 0 ? (
                <p>No cases found. Submit a New Analysis to get started.</p>
              ) : (
                (dashboardData?.cases || []).map((c) => (
                  <article key={c.case_id} className="stat-card case-card">
                    <div className="case-header">
                      <strong>Case #{c.case_id}</strong>
                      {c.risk_level && (
                        <span className={`risk-badge risk-${c.risk_level.toLowerCase().replace(/\s+/g, "-")}`}>
                          {c.risk_level}
                        </span>
                      )}
                    </div>
                    <p><strong>Date:</strong> {c.created_at}</p>
                    {c.affected_body_part && <p><strong>Body Part:</strong> {c.affected_body_part}</p>}
                    {c.condition_duration && <p><strong>Duration:</strong> {c.condition_duration}</p>}
                    {c.symptoms?.length > 0 && <p><strong>Symptoms:</strong> {c.symptoms.join(", ")}</p>}
                    {c.disease_name && (
                      <p>
                        <strong>Prediction:</strong> {c.disease_name} ({c.predicted_class}) —{" "}
                        <strong>{c.confidence}%</strong> confidence
                      </p>
                    )}
                    <p><strong>Status:</strong> <span style={{ textTransform: "capitalize" }}>{c.status || "pending"}</span></p>
                    {c.clinical_notes && (
                      <div className="clinical-recommendation-box">
                        <strong>Clinician Notes:</strong>
                        <p>{c.clinical_notes}</p>
                      </div>
                    )}
                    {c.image_url && (
                      <img src={c.image_url} alt="Lesion"
                        style={{ maxWidth: "120px", borderRadius: "8px", marginTop: "8px" }} />
                    )}
                  </article>
                ))
              )}
            </section>
          )}

        {/* ---- USER: Image Results (images prominently shown) ---- */}
        {activePage === "Image Results" && user?.role === "user" && (
          <section className="info-panel">
            <h3>Image Results</h3>
            <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
              {dashboardLoading ? "Loading…" : "Refresh"}
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            {(dashboardData?.cases || []).filter(c => c.image_url).length === 0 ? (
              <p>No images found. Upload a skin lesion image in New Analysis.</p>
            ) : (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "16px" }}>
                {(dashboardData?.cases || []).filter(c => c.image_url).map((c) => (
                  <div key={c.case_id} className="stat-card" style={{ width: "180px", textAlign: "center" }}>
                    <img src={c.image_url} alt={`Case ${c.case_id}`}
                      style={{ width: "100%", height: "140px", objectFit: "cover", borderRadius: "8px" }} />
                    <p style={{ margin: "6px 0 2px", fontWeight: 600, fontSize: "0.85rem" }}>
                      {c.disease_name || c.predicted_class || "—"}
                    </p>
                    {c.risk_level && (
                      <span className={`risk-badge risk-${c.risk_level.toLowerCase().replace(/\s+/g, "-")}`}
                        style={{ fontSize: "0.75rem" }}>
                        {c.risk_level}
                      </span>
                    )}
                    <p style={{ fontSize: "0.78rem", color: "#6b7c7a", marginTop: "4px" }}>
                      {c.confidence ? `${c.confidence}% confidence` : ""}
                    </p>
                    <p style={{ fontSize: "0.75rem", color: "#9bb0ac" }}>{c.created_at}</p>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* ---- USER: Prediction History (chronological list) ---- */}
        {activePage === "Prediction History" && user?.role === "user" && (
          <section className="info-panel">
            <h3>Prediction History</h3>
            <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
              {dashboardLoading ? "Loading…" : "Refresh"}
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            {(dashboardData?.cases || []).length === 0 ? (
              <p>No prediction history yet.</p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
                <thead>
                  <tr style={{ background: "#eef7f5", textAlign: "left" }}>
                    <th style={{ padding: "8px 12px" }}>Case #</th>
                    <th style={{ padding: "8px 12px" }}>Date</th>
                    <th style={{ padding: "8px 12px" }}>Disease</th>
                    <th style={{ padding: "8px 12px" }}>Confidence</th>
                    <th style={{ padding: "8px 12px" }}>Risk</th>
                    <th style={{ padding: "8px 12px" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(dashboardData?.cases || []).map((c, i) => (
                    <tr key={c.case_id} style={{ background: i % 2 === 0 ? "#fff" : "#f8fbfa", borderBottom: "1px solid #e2eeeb" }}>
                      <td style={{ padding: "8px 12px" }}>#{c.case_id}</td>
                      <td style={{ padding: "8px 12px" }}>{c.created_at}</td>
                      <td style={{ padding: "8px 12px" }}>{c.disease_name || c.predicted_class || "—"}</td>
                      <td style={{ padding: "8px 12px" }}>{c.confidence ? `${c.confidence}%` : "—"}</td>
                      <td style={{ padding: "8px 12px" }}>
                        {c.risk_level
                          ? <span className={`risk-badge risk-${c.risk_level.toLowerCase().replace(/\s+/g, "-")}`} style={{ fontSize: "0.75rem" }}>{c.risk_level}</span>
                          : "—"}
                      </td>
                      <td style={{ padding: "8px 12px", textTransform: "capitalize" }}>{c.status || "pending"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )}

        {/* ---- USER: Reports (PDF, CSV, JSON export per case) ---- */}
        {activePage === "Reports" && user?.role === "user" && (
          <section className="info-panel">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "14px", marginBottom: "16px" }}>
              <div>
                <h3>Analysis & Clinical Reports</h3>
                <p style={{ margin: "4px 0 0 0", color: "#5a7a76", fontSize: "0.9rem" }}>
                  Download formal medical screening reports for individual cases or export your entire diagnostic history.
                </p>
              </div>
              <div style={{ display: "flex", gap: "10px" }}>
                <button
                  type="button"
                  className="btn-export-all-csv"
                  onClick={() => exportAllCasesCSV(dashboardData?.cases || [])}
                  disabled={(dashboardData?.cases || []).length === 0}
                >
                  📥 Export All Cases (CSV)
                </button>
                <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
                  {dashboardLoading ? "Loading…" : "Refresh"}
                </button>
              </div>
            </div>

            {message && <p className="upload-note" role="status">{message}</p>}
            {(dashboardData?.cases || []).length === 0 ? (
              <p>No cases found. Submit a New Analysis to generate a report.</p>
            ) : (
              (dashboardData?.cases || []).map((c) => (
                <article key={c.case_id} className="stat-card case-card">
                  <div className="case-header">
                    <strong>Case #{c.case_id} — {c.disease_name || c.predicted_class || "No prediction"}</strong>
                    {c.risk_level && (
                      <span className={`risk-badge risk-${c.risk_level.toLowerCase().replace(/\s+/g, "-")}`}>
                        {c.risk_level}
                      </span>
                    )}
                  </div>
                  <p><strong>Date:</strong> {c.created_at}</p>
                  {c.affected_body_part && <p><strong>Body Part:</strong> {c.affected_body_part}</p>}
                  {c.disease_name && <p><strong>Prediction:</strong> {c.disease_name} — {c.confidence}% confidence</p>}
                  {c.image_url && (
                    <img src={c.image_url} alt="Lesion thumbnail" style={{ maxWidth: "140px", borderRadius: "8px", margin: "10px 0" }} />
                  )}

                  {/* Section 15: PDF, CSV, JSON Export Buttons */}
                  <div className="report-card-actions">
                    <button
                      type="button"
                      className="btn-export-pdf"
                      onClick={() => exportCasePDF(c, c.image_url)}
                    >
                      📄 Download PDF Report
                    </button>
                    <button
                      type="button"
                      className="btn-export-csv"
                      onClick={() => exportCaseCSV(c)}
                    >
                      📊 Download CSV
                    </button>
                    <button
                      type="button"
                      className="btn-export-json"
                      onClick={() => exportCaseJSON(c)}
                    >
                      💾 Download JSON
                    </button>
                    <button
                      type="button"
                      className="btn-export-email"
                      onClick={() => handleResendEmailReport(c.case_id)}
                      disabled={isSendingEmail}
                    >
                      ✉️ Email Alert
                    </button>
                  </div>
                </article>
              ))
            )}
          </section>
        )}

        {/* ============================================================
             SECTION 17: NOTIFICATION & ALERT CENTER (All Roles)
             ============================================================ */}
        {activePage === "Notifications & Alerts" && (
          <section className="info-panel notif-center-panel">
            <div className="notif-center-header">
              <div>
                <span className="eyebrow" style={{ color: "#148074" }}>SYSTEM SURVEILLANCE & TRIAGE</span>
                <h3>Clinical Notification & Alert Center</h3>
                <p>
                  Surveillance alerts for AI analysis completion, quality warnings, low-confidence predictions,
                  malignant high-risk detections, and scheduled patient follow-up reminders.
                </p>
              </div>
              <div className="notif-center-controls">
                <button type="button" className="btn-mark-all" onClick={markAllNotificationsRead}>
                  ✓ Mark All Read
                </button>
              </div>
            </div>

            {/* Severity Filter Tabs */}
            <div className="notif-filter-tabs">
              {[
                ["all", `All Alerts (${notifications.length})`],
                ["Critical", `🚨 Critical (${notifications.filter((n) => n.level === "Critical").length})`],
                ["High", `⚠️ High Risk (${notifications.filter((n) => n.level === "High").length})`],
                ["Medium", `⚡ Medium (${notifications.filter((n) => n.level === "Medium").length})`],
                ["Informational", `✓ Info (${notifications.filter((n) => n.level === "Informational").length})`],
                ["Low", `Low / Follow-ups (${notifications.filter((n) => n.level === "Low").length})`],
              ].map(([filterKey, filterLabel]) => (
                <button
                  key={filterKey}
                  type="button"
                  className={`notif-filter-btn ${notifFilter === filterKey ? "filter-active" : ""}`}
                  onClick={() => setNotifFilter(filterKey)}
                >
                  {filterLabel}
                </button>
              ))}
            </div>

            <div className="notif-alerts-grid">
              {notifications
                .filter((n) => notifFilter === "all" || n.level === notifFilter)
                .length === 0 ? (
                <p className="notif-empty">No alerts found under this severity category.</p>
              ) : (
                notifications
                  .filter((n) => notifFilter === "all" || n.level === notifFilter)
                  .map((n) => (
                    <article key={n.id} className={`alert-card alert-level-${n.level.toLowerCase()} ${n.read ? "alert-read" : "alert-unread"}`}>
                      <div className="alert-card-header">
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <span className={`alert-level-badge badge-${n.level.toLowerCase()}`}>
                            {n.level === "Critical" && "🚨 CRITICAL"}
                            {n.level === "High" && "⚠️ HIGH RISK"}
                            {n.level === "Medium" && "⚡ MEDIUM RISK"}
                            {n.level === "Low" && "ℹ️ LOW / REMINDER"}
                            {n.level === "Informational" && "✓ INFORMATIONAL"}
                          </span>
                          <strong className="alert-card-title">{n.title}</strong>
                        </div>
                        <span className="alert-time">{n.timestamp}</span>
                      </div>
                      <p className="alert-card-message">{n.message}</p>
                      <div className="alert-card-footer">
                        <span className="alert-protocol-tag">Protocol: Section 17 Clinical Decision Support</span>
                        <div style={{ display: "flex", gap: "8px" }}>
                          {!n.read && (
                            <button
                              type="button"
                              className="btn-alert-action"
                              onClick={() => setNotifications((prev) => prev.map((item) => (item.id === n.id ? { ...item, read: true } : item)))}
                            >
                              Mark Read
                            </button>
                          )}
                          <button
                            type="button"
                            className="btn-alert-dismiss"
                            onClick={() => clearNotification(n.id)}
                          >
                            Dismiss
                          </button>
                        </div>
                      </div>
                    </article>
                  ))
              )}
            </div>

            <div className="alert-clinical-disclaimer">
              <strong>Controlled Notification Terminology:</strong> In compliance with clinical medical AI guidelines,
              alert terminology is restricted to clinical decision-support indicators. Predictions do not substitute for a physical histopathological examination.
            </div>
          </section>
        )}

        {/* ============================================================
             CLINICIAN PAGES
             ============================================================ */}

        {/* Clinician: Total Cases & Review Queue */}
        {["Total Cases", "Cases Requiring Review", "Recent Cases"].includes(activePage) &&
          user?.role === "clinician" && (
            <section className="info-panel">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px", flexWrap: "wrap", gap: "12px" }}>
                <div>
                  <h3 style={{ margin: "0 0 4px 0" }}>Clinical Cases Registry & Review Queue</h3>
                  <p style={{ margin: 0, color: "#64748b", fontSize: "0.92rem" }}>
                    Manage assigned patient lesion cases, inspect multimodal predictions, and record clinical diagnostic notes.
                  </p>
                </div>
                <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
                  {dashboardLoading ? "Loading…" : "Refresh"}
                </button>
              </div>

              {/* Intuitive Filter Tabs */}
              <div style={{ display: "flex", gap: "8px", marginBottom: "18px" }}>
                <button
                  type="button"
                  style={{
                    background: clinicianCaseFilter === "all" ? "#0d9488" : "#f1f5f9",
                    color: clinicianCaseFilter === "all" ? "#ffffff" : "#334155",
                    border: "1px solid",
                    borderColor: clinicianCaseFilter === "all" ? "#0d9488" : "#cbd5e1",
                    padding: "7px 14px",
                    borderRadius: "8px",
                    fontWeight: 600,
                    fontSize: "0.85rem",
                    cursor: "pointer",
                  }}
                  onClick={() => setClinicianCaseFilter("all")}
                >
                  All Cases ({(dashboardData?.cases || []).length})
                </button>
                <button
                  type="button"
                  style={{
                    background: clinicianCaseFilter === "pending" ? "#0d9488" : "#f1f5f9",
                    color: clinicianCaseFilter === "pending" ? "#ffffff" : "#334155",
                    border: "1px solid",
                    borderColor: clinicianCaseFilter === "pending" ? "#0d9488" : "#cbd5e1",
                    padding: "7px 14px",
                    borderRadius: "8px",
                    fontWeight: 600,
                    fontSize: "0.85rem",
                    cursor: "pointer",
                  }}
                  onClick={() => setClinicianCaseFilter("pending")}
                >
                  Pending Review ({(dashboardData?.cases || []).filter((c) => c.status === "pending").length})
                </button>
                <button
                  type="button"
                  style={{
                    background: clinicianCaseFilter === "reviewed" ? "#0d9488" : "#f1f5f9",
                    color: clinicianCaseFilter === "reviewed" ? "#ffffff" : "#334155",
                    border: "1px solid",
                    borderColor: clinicianCaseFilter === "reviewed" ? "#0d9488" : "#cbd5e1",
                    padding: "7px 14px",
                    borderRadius: "8px",
                    fontWeight: 600,
                    fontSize: "0.85rem",
                    cursor: "pointer",
                  }}
                  onClick={() => setClinicianCaseFilter("reviewed")}
                >
                  Reviewed ({(dashboardData?.cases || []).filter((c) => c.status === "reviewed").length})
                </button>
              </div>

              {message && <p className="upload-note" role="status">{message}</p>}
              {(dashboardData?.cases || []).length === 0 ? (
                <p>No cases found.</p>
              ) : (
                (dashboardData?.cases || [])
                  .filter((c) => {
                    if (clinicianCaseFilter === "pending") return c.status === "pending";
                    if (clinicianCaseFilter === "reviewed") return c.status === "reviewed";
                    return true;
                  })
                  .map((c) => (
                    <article key={c.case_id} className="stat-card case-card">
                      <div className="case-header">
                        <strong>
                          {c.patient_name || "Patient"}{c.patient_age ? `, Age ${c.patient_age}` : ""}{c.patient_gender ? `, ${c.patient_gender}` : ""}
                        </strong>
                        {c.risk_level && (
                          <span className={`risk-badge risk-${c.risk_level.toLowerCase().replace(/\s+/g, "-")}`}>
                            {c.risk_level}
                          </span>
                        )}
                      </div>
                      <p>
                        <strong>Date:</strong> {c.created_at} &nbsp;|&nbsp;
                        <strong>Status:</strong>{" "}
                        <span style={{ textTransform: "capitalize" }}>{c.status || "pending"}</span>
                      </p>
                      {c.affected_body_part && (
                        <p><strong>Body Part:</strong> {c.affected_body_part}</p>
                      )}
                      {c.symptoms?.length > 0 && (
                        <p><strong>Symptoms:</strong> {c.symptoms.join(", ")}</p>
                      )}
                      {c.disease_name && (
                        <p>
                          <strong>AI Prediction:</strong> {c.disease_name} ({c.predicted_class}) —{" "}
                          <strong>{c.confidence}%</strong>
                        </p>
                      )}
                      {c.image_url && (
                        <img
                          src={c.image_url}
                          alt="Lesion"
                          style={{ maxWidth: "120px", borderRadius: "8px", marginTop: "8px" }}
                        />
                      )}
                      {c.clinical_notes && (
                        <div className="clinical-recommendation-box">
                          <strong>Previous Notes:</strong>
                          <p>{c.clinical_notes}</p>
                        </div>
                      )}

                      {/* Review Form */}
                      {selectedCase === c.case_id ? (
                        <div className="review-form">
                          <label>
                            Status:
                            <select
                              value={reviewStatus}
                              onChange={(e) => setReviewStatus(e.target.value)}
                            >
                              <option value="reviewed">Reviewed</option>
                              <option value="pending">Keep Pending</option>
                              <option value="urgent">Urgent</option>
                            </select>
                          </label>
                          <label>
                            Clinical Notes:
                            <textarea
                              value={reviewNotes}
                              onChange={(e) => setReviewNotes(e.target.value)}
                              placeholder="Add clinical notes…"
                              rows={3}
                            />
                          </label>
                          <div style={{ display: "flex", gap: "8px" }}>
                            <button type="button" onClick={() => submitClinicianReview(c.case_id)}>
                              Submit Review
                            </button>
                            <button
                              type="button"
                              className="text-button"
                              onClick={() => { setSelectedCase(null); setReviewNotes(""); }}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <button
                          type="button"
                          style={{ marginTop: "8px" }}
                          onClick={() => {
                            setSelectedCase(c.case_id);
                            setReviewNotes(c.clinical_notes || "");
                            setReviewStatus("reviewed");
                          }}
                        >
                          Review Case
                        </button>
                      )}
                    </article>
                  ))
              )}
            </section>
          )}

        {/* Clinician: Disease Distribution */}
        {activePage === "Disease Distribution" && user?.role === "clinician" && (
          <section className="info-panel">
            <h3>Disease Distribution</h3>
            <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
              {dashboardLoading ? "Loading…" : "Refresh"}
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            {!dashboardData ? (
              <p>Loading…</p>
            ) : Object.keys(dashboardData?.disease_distribution || {}).length === 0 ? (
              <p>No cases recorded yet.</p>
            ) : (
              <div className="prob-bars-list">
                {Object.entries(dashboardData.disease_distribution)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, count]) => {
                    const total = Object.values(dashboardData.disease_distribution).reduce(
                      (s, v) => s + v, 0
                    );
                    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                    return (
                      <div key={cls} className="prob-bar-row">
                        <div className="prob-labels">
                          <span>
                            <strong>{cls}</strong> — {DISEASE_NAMES_MAP[cls] || cls}
                          </span>
                          <span>{count} case{count !== 1 ? "s" : ""} ({pct}%)</span>
                        </div>
                        <div className="prob-bar-track">
                          <div className="prob-bar-fill" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
              </div>
            )}
          </section>
        )}

        {/* Clinician: Prediction Confidence */}
        {activePage === "Prediction Confidence" && user?.role === "clinician" && (
          <section className="info-panel">
            <h3>Prediction Confidence Overview</h3>
            <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
              {dashboardLoading ? "Loading…" : "Refresh"}
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            {!dashboardData ? (
              <p>Loading…</p>
            ) : (
              <>
                {(() => {
                  const cases = dashboardData?.cases || [];
                  const withConf = cases.filter((c) => c.confidence != null);
                  const avg =
                    withConf.length > 0
                      ? (withConf.reduce((s, c) => s + parseFloat(c.confidence), 0) / withConf.length).toFixed(1)
                      : "—";
                  const highConf = withConf.filter((c) => parseFloat(c.confidence) >= 80);
                  return (
                    <>
                      <section className="stats-grid" style={{ marginBottom: "16px" }}>
                        <article className="stat-card">
                          <span>Average Confidence</span>
                          <h2>{avg}{avg !== "—" ? "%" : ""}</h2>
                          <p>Across all cases</p>
                        </article>
                        <article className="stat-card">
                          <span>High Confidence</span>
                          <h2>{highConf.length}</h2>
                          <p>≥80% confidence</p>
                        </article>
                        <article className="stat-card">
                          <span>Total Cases</span>
                          <h2>{withConf.length}</h2>
                          <p>With predictions</p>
                        </article>
                      </section>
                      {withConf.sort((a, b) => parseFloat(b.confidence) - parseFloat(a.confidence)).map((c) => (
                        <article key={c.case_id} className="stat-card case-card" style={{ marginBottom: "8px" }}>
                          <div className="case-header">
                            <strong>Case #{c.case_id} — {c.disease_name || c.predicted_class}</strong>
                            <span>{c.confidence}% confidence</span>
                          </div>
                          <div className="prob-bar-track">
                            <div
                              className="prob-bar-fill"
                              style={{ width: `${c.confidence}%`, background: parseFloat(c.confidence) >= 80 ? "#2ecc71" : parseFloat(c.confidence) >= 60 ? "#f39c12" : "#e74c3c" }}
                            />
                          </div>
                        </article>
                      ))}
                    </>
                  );
                })()}
              </>
            )}
          </section>
        )}

        {/* ML Analytics & Model Benchmark Dashboard (Admin & Clinician) */}
        {((activePage === "ML Analytics" && user?.role === "admin") ||
          (activePage === "Model Performance" && user?.role === "clinician")) && (
          <section className="info-panel ml-structured-dashboard">
            {/* Executive Hero Banner */}
            <div className="ml-overview-hero">
              <span className="eyebrow" style={{ color: "#a7f3d0" }}>CLINICAL MACHINE LEARNING ANALYTICS</span>
              <h3>Multimodal AI Architecture Benchmark & Evaluation Suite</h3>
              <p>
                Empirical evaluation of four multimodal convolutional architectures fused with 64-dimensional clinical metadata
                on the PAD-UFES-20 dataset. Evaluated on 460 held-out patient cases with zero data leakage.
              </p>

              <div className="ml-stat-ribbon">
                <div className="ml-ribbon-stat highlight">
                  <span>1. Train Accuracy</span>
                  <strong>98.07%</strong>
                  <small>Multimodal Convergence</small>
                </div>
                <div className="ml-ribbon-stat highlight">
                  <span>2. Validation Accuracy</span>
                  <strong>77.55%</strong>
                  <small>Cross-Validation Peak</small>
                </div>
                <div className="ml-ribbon-stat highlight">
                  <span>3. Test Accuracy</span>
                  <strong>76.74%</strong>
                  <small>PAD-UFES-20 Holdout</small>
                </div>
                <div className="ml-ribbon-stat">
                  <span>4. Clinical Cases</span>
                  <strong>2,298</strong>
                  <small>6 Classes · Multimodal Fusion</small>
                </div>
                <div className="ml-ribbon-stat">
                  <span>Macro Precision</span>
                  <strong>73.51%</strong>
                  <small>Positive Predictive Value</small>
                </div>
                <div className="ml-ribbon-stat">
                  <span>Macro F1-Score</span>
                  <strong>71.86%</strong>
                  <small>Harmonic Balanced</small>
                </div>
              </div>
            </div>

            {/* Sub-Navigation Filter Tabs */}
            <nav className="ml-subnav-bar" aria-label="ML Analytics Navigation">
              {[
                ["all", "📑 Complete Dossier"],
                ["comparison", "📊 1.0 Benchmark & Decision Matrix"],
                ["charts", "📈 2.0 Performance Visualizer"],
                ["dataset", "🗃️ 3.0 Dataset Analytics"],
                ["architectures", "🔬 4.0 Architecture Deep-Dives"],
              ].map(([subKey, subLabel]) => (
                <button
                  key={subKey}
                  type="button"
                  className={`ml-subnav-btn ${mlAnalyticsSubSection === subKey ? "subnav-active" : ""}`}
                  onClick={() => setMlAnalyticsSubSection(subKey)}
                >
                  {subLabel}
                </button>
              ))}
            </nav>

            {/* ========================================================
                SECTION 1.0: BENCHMARK COMPARISON & DECISION MATRIX
                ======================================================== */}
            {(mlAnalyticsSubSection === "all" || mlAnalyticsSubSection === "comparison") && (
              <div className="ml-section-box">
                <div className="ml-section-header-row">
                  <div className="ml-section-title-wrap">
                    <span className="ml-section-num">1.0</span>
                    <div>
                      <h4>Multimodal CNN Benchmark & Clinical Decision Matrix</h4>
                      <p className="ml-section-subtitle">
                        Rigorous comparison across 460 held-out test patients evaluated after 12 identical epochs.
                      </p>
                    </div>
                  </div>
                  <span className="rank-chip-gold">
                    ★ Production Model: EfficientNet-B0
                  </span>
                </div>

                <div className="ml-table-wrapper">
                  <table className="ml-comparison-table">
                    <thead>
                      <tr>
                        <th>Architecture</th>
                        <th>Test Accuracy</th>
                        <th>Macro Precision</th>
                        <th>Macro Recall (Sensitivity)</th>
                        <th>Macro F1 (Balanced)</th>
                        <th>Training Time (12 ep)</th>
                        <th>Inference Latency</th>
                        <th>Clinical Recommendation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {CNN_MODEL_COMPARISON.map((m, idx) => (
                        <tr key={m.id} className={m.isProduction ? "row-selected" : ""}>
                          <td>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              {m.isProduction ? (
                                <span className="rank-chip-gold">🥇 1st</span>
                              ) : m.accuracy === 77.83 ? (
                                <span className="rank-chip-silver">🥈 2nd</span>
                              ) : (
                                <span className="rank-chip-standard">{idx + 1}th</span>
                              )}
                              <strong>{m.name}</strong>
                            </div>
                            <small style={{ color: "#657e79", fontSize: "0.75rem" }}>
                              {m.params} params • {m.featureDim}d features
                            </small>
                          </td>
                          <td>
                            <strong style={{ color: m.isProduction ? "#148074" : "inherit" }}>
                              {m.accuracy}%
                            </strong>
                            {m.isProduction && <span style={{ marginLeft: "4px", fontSize: "0.7rem", color: "#148074", fontWeight: 700 }}>[Selected]</span>}
                            {!m.isProduction && m.accuracy === 77.83 && <span style={{ marginLeft: "4px", fontSize: "0.7rem", color: "#64748b", fontWeight: 500 }}>(Majority bias)</span>}
                          </td>
                          <td>{m.precision}%</td>
                          <td>
                            <strong style={{ color: m.recall === 70.47 ? "#148074" : "inherit" }}>
                              {m.recall}%
                            </strong>
                            {m.recall === 70.47 && <span style={{ marginLeft: "4px", fontSize: "0.7rem", color: "#148074", fontWeight: 700 }}>[Best Sensitivity]</span>}
                          </td>
                          <td>
                            <strong style={{ color: m.macroF1 === 71.86 ? "#148074" : "inherit" }}>
                              {m.macroF1}%
                            </strong>
                            {m.macroF1 === 71.86 && <span style={{ marginLeft: "4px", fontSize: "0.7rem", color: "#148074", fontWeight: 700 }}>[Best F1]</span>}
                          </td>
                          <td>{m.trainTime} s</td>
                          <td>
                            <strong style={{ color: m.infTime === 34.09 ? "#148074" : "inherit" }}>
                              {m.infTime} ms
                            </strong>
                            {m.infTime === 34.09 && <span style={{ marginLeft: "4px", fontSize: "0.7rem", color: "#148074", fontWeight: 700 }}>[Fastest]</span>}
                          </td>
                          <td>
                            <span className={m.isProduction ? "model-prod-tag" : "model-highlight-badge"}>
                              {m.badge}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Structured Clinical Decision Breakdown */}
                <h5 style={{ margin: "20px 0 6px 0", color: "#0b554d", fontSize: "1rem" }}>
                  Clinical Selection Framework: Why EfficientNet-B0 Over ResNet50?
                </h5>
                <div className="decision-matrix-grid">
                  <div className="decision-card">
                    <span className="decision-card-icon">⚖️</span>
                    <h5>1. The Accuracy Paradox in Dermatology</h5>
                    <p>
                      ResNet50 scored higher overall accuracy (77.83% vs 76.74%), but 69% of the dataset is dominated by common BCC and ACK.
                      Models biased toward majority classes artificially inflate accuracy while misclassifying minority classes.
                    </p>
                  </div>
                  <div className="decision-card" style={{ borderLeft: "4px solid #148074" }}>
                    <span className="decision-card-icon">🩺</span>
                    <h5>2. Sensitivity on Malignant Cancers</h5>
                    <p>
                      EfficientNet-B0 achieved <strong>70.47% Macro Recall</strong> and <strong>71.86% Macro F1</strong> (vs 64.13% recall and 67.47% F1 for ResNet50).
                      Missing a deadly Melanoma (MEL) or Squamous Cell Carcinoma (SCC) is catastrophic in healthcare; high recall saves lives.
                    </p>
                  </div>
                  <div className="decision-card">
                    <span className="decision-card-icon">⚡</span>
                    <h5>3. Edge & Mobile Feasibility</h5>
                    <p>
                      EfficientNet-B0 delivers the lowest inference latency (<strong>34.09 ms/image</strong>) and a compact <strong>~5.3M parameter</strong> footprint,
                      enabling smooth real-time execution on mobile devices with native Grad-CAM explainability.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* ========================================================
                SECTION 2.0: INTERACTIVE PERFORMANCE VISUALIZER
                ======================================================== */}
            {(mlAnalyticsSubSection === "all" || mlAnalyticsSubSection === "charts") && (
              <div className="ml-section-box">
                <div className="ml-section-header-row">
                  <div className="ml-section-title-wrap">
                    <span className="ml-section-num">2.0</span>
                    <div>
                      <h4>Interactive Comparative Performance Visualizer</h4>
                      <p className="ml-section-subtitle">
                        Dynamic metric visualizer and saved high-resolution comparative figures across all architectures.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="charts-tabs-header">
                  {[
                    ["accuracy", "Accuracy (%)"],
                    ["f1", "Macro F1 (%)"],
                    ["recall", "Macro Recall (%)"],
                    ["precision", "Macro Precision (%)"],
                    ["trainTime", "Training Time (s)"],
                    ["infTime", "Inference Latency (ms)"],
                    ["staticPlots", "Saved High-Res Figures"],
                  ].map(([tabKey, tabLabel]) => (
                    <button
                      key={tabKey}
                      type="button"
                      className={`chart-tab-btn ${activeChartTab === tabKey ? "tab-active" : ""}`}
                      onClick={() => setActiveChartTab(tabKey)}
                    >
                      {tabLabel}
                    </button>
                  ))}
                </div>

                {activeChartTab !== "staticPlots" ? (
                  <div className="interactive-bars-list">
                    {CNN_MODEL_COMPARISON.map((m) => {
                      let val = 0;
                      let displayVal = "";
                      let barPct = 0;

                      if (activeChartTab === "accuracy") {
                        val = m.accuracy;
                        displayVal = `${val}%`;
                        barPct = ((val - 65) / 15) * 100;
                      } else if (activeChartTab === "f1") {
                        val = m.macroF1;
                        displayVal = `${val}%`;
                        barPct = ((val - 62) / 12) * 100;
                      } else if (activeChartTab === "recall") {
                        val = m.recall;
                        displayVal = `${val}%`;
                        barPct = ((val - 60) / 13) * 100;
                      } else if (activeChartTab === "precision") {
                        val = m.precision;
                        displayVal = `${val}%`;
                        barPct = ((val - 62) / 15) * 100;
                      } else if (activeChartTab === "trainTime") {
                        val = m.trainTime;
                        displayVal = `${val} s`;
                        barPct = ((val - 880) / 90) * 100;
                      } else if (activeChartTab === "infTime") {
                        val = m.infTime;
                        displayVal = `${val} ms`;
                        barPct = ((val - 32) / 6) * 100;
                      }

                      return (
                        <div key={m.id} className="interactive-bar-row">
                          <div className="bar-meta-row">
                            <div className="bar-meta-left">
                              <strong>{m.name}</strong> {m.isProduction && <span className="model-prod-tag">DEPLOYED MODEL</span>}
                            </div>
                            <div className="bar-meta-right" style={{ color: m.color }}>
                              {displayVal}
                            </div>
                          </div>
                          <div className="bar-outer-track">
                            <div
                              className="bar-inner-fill"
                              style={{
                                width: `${Math.min(Math.max(barPct, 8), 100)}%`,
                                background: m.color,
                              }}
                            >
                              {displayVal}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="static-plots-grid">
                    <div className="static-plot-card">
                      <h5>Multimodal CNN Evaluation Metrics Bar Chart</h5>
                      <img
                        src="/DermAI_CNN_Comparison_BarChart.png"
                        alt="CNN Architecture Comparison Bar Chart"
                        className="static-plot-img"
                      />
                    </div>
                    <div className="static-plot-card">
                      <h5>Training & Inference Computational Latency</h5>
                      <img
                        src="/DermAI_CNN_Inference_Speed.png"
                        alt="Inference Speed and Training Time"
                        className="static-plot-img"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ========================================================
                SECTION 3.0: PAD-UFES-20 DATASET ANALYTICS
                ======================================================== */}
            {(mlAnalyticsSubSection === "all" || mlAnalyticsSubSection === "dataset") && (
              <div className="ml-section-box">
                <div className="ml-section-header-row">
                  <div className="ml-section-title-wrap">
                    <span className="ml-section-num">3.0</span>
                    <div>
                      <h4>PAD-UFES-20 Clinical Dataset Analytics & Cohort Distribution</h4>
                      <p className="ml-section-subtitle">
                        Comprehensive class balance breakdown, patient metadata volume, and data leakage safeguards.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="dataset-summary-cards">
                  <div className="dataset-subcard">
                    <span>Total Images</span>
                    <strong>{DATASET_ANALYTICS.totalImages}</strong>
                  </div>
                  <div className="dataset-subcard">
                    <span>Unique Patients</span>
                    <strong>{DATASET_ANALYTICS.totalPatients}</strong>
                  </div>
                  <div className="dataset-subcard">
                    <span>Diagnostic Classes</span>
                    <strong>{DATASET_ANALYTICS.categoriesCount}</strong>
                  </div>
                  <div className="dataset-subcard">
                    <span>Matched Records</span>
                    <strong>{DATASET_ANALYTICS.matchedImages}</strong>
                  </div>
                  <div className="dataset-subcard">
                    <span>Missing Matches</span>
                    <strong>{DATASET_ANALYTICS.missingMatches}</strong>
                  </div>
                  <div className="dataset-subcard">
                    <span>Duplicate Records</span>
                    <strong>{DATASET_ANALYTICS.duplicateRecords}</strong>
                  </div>
                </div>

                <div className="distribution-columns">
                  {/* Disease Class Imbalance */}
                  <div className="dist-column-card">
                    <h5>Disease Class Imbalance (2,298 Images)</h5>
                    {DATASET_ANALYTICS.diseaseDistribution.map((d) => (
                      <div key={d.code} className="dist-bar-item">
                        <div className="dist-bar-item-header">
                          <span>
                            <strong>{d.code}</strong> — {d.name} ({d.risk})
                          </span>
                          <strong>{d.count} ({d.pct}%)</strong>
                        </div>
                        <div className="bar-outer-track" style={{ height: "14px" }}>
                          <div
                            className="bar-inner-fill"
                            style={{ width: `${d.pct}%`, background: d.color }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Patient-Wise Partitioning */}
                  <div className="dist-column-card">
                    <h5>Patient-Wise Stratified Partitioning</h5>
                    {DATASET_ANALYTICS.splits.map((s) => (
                      <div key={s.name} className="dist-bar-item">
                        <div className="dist-bar-item-header">
                          <span><strong>{s.name}</strong></span>
                          <strong>{s.count} ({s.pct}%)</strong>
                        </div>
                        <div className="bar-outer-track" style={{ height: "14px" }}>
                          <div
                            className="bar-inner-fill"
                            style={{ width: `${s.pct}%`, background: s.color }}
                          ></div>
                        </div>
                        <p style={{ margin: "4px 0 10px 0", fontSize: "0.75rem", color: "#657e79" }}>
                          {s.note}
                        </p>
                      </div>
                    ))}
                    <div className="leakage-guarantee-box">
                      <strong>✓ Zero Data Leakage Guarantee:</strong> Patient ID stratification ensures that multiple photos from the same patient never appear in both training and testing sets.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ========================================================
                SECTION 4.0: ARCHITECTURAL DEEP-DIVE SPECIFICATIONS
                ======================================================== */}
            {(mlAnalyticsSubSection === "all" || mlAnalyticsSubSection === "architectures") && (
              <div className="ml-section-box">
                <div className="ml-section-header-row">
                  <div className="ml-section-title-wrap">
                    <span className="ml-section-num">4.0</span>
                    <div>
                      <h4>Technical Deep-Dive & Architecture Specifications</h4>
                      <p className="ml-section-subtitle">
                        In-depth architectural composition, tensor dimensionalities, and multimodal fusion mechanics.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="model-deep-dive-grid">
                  {CNN_MODEL_COMPARISON.map((m) => (
                    <div key={m.id} className={`deep-dive-card ${m.isProduction ? "card-production" : ""}`}>
                      <div>
                        <div className="deep-dive-header">
                          <div>
                            <h4>{m.name}</h4>
                            <small style={{ color: "#657e79", fontSize: "0.78rem" }}>{m.displayName}</small>
                          </div>
                          <span className={m.isProduction ? "model-prod-tag" : "model-highlight-badge"}>
                            {m.badge}
                          </span>
                        </div>
                        <p className="deep-dive-desc">{m.architecture}</p>

                        <div className="deep-dive-metrics">
                          <div className="deep-metric-item">
                            <span>Accuracy</span>
                            <strong>{m.accuracy}%</strong>
                          </div>
                          <div className="deep-metric-item">
                            <span>Macro F1</span>
                            <strong>{m.macroF1}%</strong>
                          </div>
                          <div className="deep-metric-item">
                            <span>Latency</span>
                            <strong>{m.infTime}ms</strong>
                          </div>
                        </div>

                        <div style={{ fontSize: "0.8rem", color: "#506f69", marginBottom: "8px" }}>
                          <strong>Parameters:</strong> {m.params} &nbsp;|&nbsp; <strong>Feature Dim:</strong> {m.featureDim} &nbsp;|&nbsp; <strong>Fused Head:</strong> 320d
                        </div>
                      </div>

                      <div className="deep-dive-pros">
                        <strong>Clinical Assessment:</strong> {m.pros}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}


        {/* ============================================================
             ADMIN PAGES
             ============================================================ */}

        {/* Admin: Registered Users */}
        {activePage === "Registered Users" && user?.role === "admin" && (
          <section className="info-panel">
            <h3>Registered Users</h3>
            <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
              {dashboardLoading ? "Loading…" : "Refresh"}
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            {(dashboardData?.users || []).length === 0 ? (
              <p>No users found.</p>
            ) : (
              (dashboardData?.users || []).map((u) => (
                <article key={u.user_id} className="stat-card" style={{ marginBottom: "12px" }}>
                  <div className="case-header">
                    <strong>{u.name}</strong>
                    <span className={`risk-badge risk-${u.role === "admin" ? "high-risk" : u.role === "clinician" ? "moderate-risk" : "low-risk"}`}>
                      {u.role}
                    </span>
                  </div>
                  <p>{u.email}</p>
                  <p>
                    <strong>Joined:</strong> {u.created_at} &nbsp;|&nbsp;
                    <strong>Approved:</strong> {u.is_approved ? "Yes" : "No"}
                  </p>
                  <label style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "8px" }}>
                    Change Role:
                    <select
                      defaultValue={u.role}
                      onChange={(e) => adminUpdateRole(u.user_id, e.target.value)}
                    >
                      <option value="user">User</option>
                      <option value="clinician">Clinician</option>
                      <option value="admin">Admin</option>
                    </select>
                  </label>
                </article>
              ))
            )}
          </section>
        )}

        {/* Admin: Total Analyses & Platform Activity */}
        {["Total Analyses", "Model Usage", "System Activity", "Prediction Statistics"].includes(activePage) &&
          user?.role === "admin" && (
            <section className="info-panel">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px", flexWrap: "wrap", gap: "12px" }}>
                <div>
                  <h3 style={{ margin: "0 0 4px 0" }}>Total Analyses & Diagnostic Volume</h3>
                  <p style={{ margin: 0, color: "#64748b", fontSize: "0.92rem" }}>
                    Platform-wide AI classification volume, active users, and real-time clinical condition breakdown.
                  </p>
                </div>
                <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
                  {dashboardLoading ? "Loading…" : "Refresh"}
                </button>
              </div>
              {message && <p className="upload-note" role="status">{message}</p>}
              {dashboardData && (
                <>
                  <section className="stats-grid" style={{ marginBottom: "24px" }}>
                    <article className="stat-card">
                      <span>Total Users</span>
                      <h2>{dashboardData.total_users ?? 0}</h2>
                      <p>Registered accounts</p>
                    </article>
                    <article className="stat-card">
                      <span>Total Analyses</span>
                      <h2>{dashboardData.total_analyses ?? 0}</h2>
                      <p>AI screenings executed</p>
                    </article>
                    <article className="stat-card">
                      <span>Pending Clinicians</span>
                      <h2>{dashboardData.pending_clinicians_count ?? 0}</h2>
                      <p>Awaiting verification</p>
                    </article>
                  </section>

                  <h4>Disease Distribution (Platform-wide)</h4>
                  {Object.keys(dashboardData?.disease_distribution || {}).length === 0 ? (
                    <p>No predictions yet.</p>
                  ) : (
                    <div className="prob-bars-list">
                      {Object.entries(dashboardData.disease_distribution)
                        .sort((a, b) => b[1] - a[1])
                        .map(([cls, count]) => {
                          const total = Object.values(dashboardData.disease_distribution).reduce(
                            (s, v) => s + v, 0
                          );
                          const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                          return (
                            <div key={cls} className="prob-bar-row">
                              <div className="prob-labels">
                                <span>
                                  <strong>{cls}</strong> — {DISEASE_NAMES_MAP[cls] || cls}
                                </span>
                                <span>{count} ({pct}%)</span>
                              </div>
                              <div className="prob-bar-track">
                                <div className="prob-bar-fill" style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </>
              )}
            </section>
          )}

        {/* Admin: Platform Health */}
        {activePage === "Platform Health" && user?.role === "admin" && (
          <section className="info-panel">
            <h3>Model & Platform Health</h3>
            <button type="button" onClick={loadDashboard} disabled={dashboardLoading}>
              {dashboardLoading ? "Loading…" : "Refresh"}
            </button>
            {message && <p className="upload-note" role="status">{message}</p>}
            <section className="stats-grid" style={{ marginBottom: "20px" }}>
              <article className="stat-card">
                <span>Model Status</span>
                <h2>{dashboardData?.model_info?.status === "loaded" ? "✓ Online" : "—"}</h2>
                <p>{dashboardData?.model_info?.name ?? "EfficientNet-B0 Fusion"}</p>
              </article>
              <article className="stat-card">
                <span>1. Train Accuracy</span>
                <h2>{dashboardData?.model_info?.train_accuracy ?? "98.07%"}</h2>
                <p>Multimodal convergence</p>
              </article>
              <article className="stat-card">
                <span>2. Val Accuracy</span>
                <h2>{dashboardData?.model_info?.val_accuracy ?? "77.55%"}</h2>
                <p>Cross-validation peak</p>
              </article>
              <article className="stat-card">
                <span>3. Test Accuracy</span>
                <h2>{dashboardData?.model_info?.test_accuracy ?? "76.74%"}</h2>
                <p>PAD-UFES-20 holdout</p>
              </article>
              <article className="stat-card">
                <span>4. Diagnostic Scope</span>
                <h2>{dashboardData?.model_info?.cases_count ?? "2,298"}</h2>
                <p>Clinical patient cases</p>
              </article>
              <article className="stat-card">
                <span>Macro F1-Score</span>
                <h2>{dashboardData?.model_info?.test_macro_f1 ?? "71.86%"}</h2>
                <p>Harmonic balanced score</p>
              </article>
            </section>
            {dashboardData?.model_info && (
              <>
                <p><strong>Classes:</strong> {(dashboardData.model_info.classes || []).join(", ")}</p>
                <p><strong>Multimodal:</strong> {dashboardData.model_info.is_multimodal ? "Yes — Image + Clinical Metadata Fusion" : "Image only"}</p>
                <p><strong>Architecture:</strong> EfficientNet-B0 backbone + metadata MLP branch + Grad-CAM XAI</p>
              </>
            )}
            <p><strong>Total Users:</strong> {dashboardData?.total_users ?? "—"}</p>
            <p><strong>Total Analyses:</strong> {dashboardData?.total_analyses ?? "—"}</p>
            <p><strong>Pending Clinician Approvals:</strong> {dashboardData?.pending_clinicians_count ?? "—"}</p>
          </section>
        )}

        {/* ============================================================
             PROFILE PAGE (all roles)
             ============================================================ */}

        {activePage === "Profile" && (
          <section className="info-panel profile-panel">
            <h3>User & Clinical Profile</h3>
            {user?.role === "user" && (
              <div className="patient-profile-details">
                <div className="profile-header-card">
                  <div className="profile-avatar-large">
                    {user?.name?.charAt(0).toUpperCase() || "P"}
                  </div>
                  <div>
                    <h4 style={{ margin: "0 0 4px 0", fontSize: "1.3rem", color: "#0b554d" }}>{user?.name || "Patient"}</h4>
                    <p style={{ margin: "0 0 6px 0", color: "#64748b", fontSize: "0.92rem" }}><strong>Email:</strong> {user?.email}</p>
                    <span className="profile-role-badge">Verified Patient Account</span>
                  </div>
                </div>

                <h4 style={{ marginTop: "24px", color: "#0b554d" }}>Clinical Information</h4>
                {patientProfile ? (
                  <div className="patient-profile-grid">
                    <p><strong>Patient Full Name:</strong> {user?.name}</p>
                    <p><strong>Registered Email:</strong> {user?.email}</p>
                    <p><strong>Patient Age:</strong> {patientProfile.age} years</p>
                    <p><strong>Biological Gender:</strong> {patientProfile.gender || "Not specified"}</p>
                    <p><strong>Relevant Medical History:</strong> {patientProfile.medical_history || "None recorded"}</p>
                    <p><strong>Previous Skin Treatments:</strong> {patientProfile.previous_skin_treatments || "None recorded"}</p>
                    <p><strong>Current Medications:</strong> {patientProfile.current_medications || "None recorded"}</p>
                    <div style={{ marginTop: "16px" }}>
                      <button type="button" onClick={editPatientProfile}>Edit Clinical Profile</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p>Patient details have not been added yet.</p>
                    <button type="button" onClick={editPatientProfile}>Add Patient Details</button>
                  </>
                )}
              </div>
            )}
            {(user?.role === "clinician" || user?.role === "admin") && (
              <div className="clinician-profile-card">
                <div className="profile-avatar-large">
                  {user?.name?.charAt(0).toUpperCase() || "C"}
                </div>
                <div>
                  <h4 style={{ margin: "0 0 6px 0", fontSize: "1.3rem", color: "#0b554d" }}>{user.name}</h4>
                  <p style={{ margin: "0 0 4px 0", color: "#64748b" }}><strong>Email:</strong> {user.email}</p>
                  <p style={{ margin: "0 0 6px 0", textTransform: "capitalize" }}><strong>Role:</strong> {user.role === "admin" ? "System Administrator" : "Certified Clinician / Dermatologist"}</p>
                  <span className="profile-role-badge">
                    {user.role === "admin" ? "🛡️ Administrator Access" : "🩺 Licensed Clinical Access"}
                  </span>
                </div>
              </div>
            )}
          </section>
        )}

        <footer className="dashboard-footer">
          DermAI provides preliminary AI-assisted analysis only.
          It does not replace professional medical diagnosis.
        </footer>
      </main>
    </div>
  );
}

export default App;
