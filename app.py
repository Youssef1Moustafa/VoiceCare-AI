# =========================================
# 📚 IMPORTS
# =========================================
import os
import re
import json
import uuid
import base64
import time
import pandas as pd
import numpy as np
import torch
import faiss
import gradio as gr
import plotly.express as px
import gspread
import joblib
from datetime import datetime, timezone
from functools import lru_cache
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
from huggingface_hub import login
from datetime import datetime
import pytz
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# =========================================
# 🔐 AUTHENTICATION & SHEET SETUP
# =========================================
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Google Sheets Credentials (using environment variable or file)
try:
    # Try to get credentials from environment variable
    import json as json_module
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT")
    if creds_json:
        creds_dict = json_module.loads(creds_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
    else:
        # Fallback to file
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "telecom-ai-agent-8c2eaf8a5421.json",
            SCOPE
        )
    gc = gspread.authorize(creds)
    SHEET_ID = "1GJPnnylIcCymLBBfBtR0QeQAvXa91OOaID7U3-0sqXg"
    spreadsheet = gc.open_by_key(SHEET_ID)
    sheet = spreadsheet.sheet1
    print("✅ Connected to Google Sheets")
except Exception as e:
    print(f"⚠️ Google Sheets connection error: {e}")
    sheet = None
    gc = None

# =========================================
# 🎨 UNIFIED CSS STYLING (Light Theme)
# =========================================
UNIFIED_CSS = """
/* =========================
   GLOBAL
========================= */
body, .gradio-container {
    background-color: #F5F7FA !important;
    color: #1a202c !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* =========================
   HEADER (Dashboard)
========================= */
.header-box {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
}
/* =========================
   TABS
========================= */
.tab-nav button {
    background-color: #E2E8F0 !important;
    color: #4A5568 !important;
    border: none !important;
    font-weight: 600;
    border-radius: 12px 12px 0 0;
    padding: 10px 18px;
}
.tab-nav button.selected {
    background-color: #ffffff !important;
    color: #008CBA !important;
    border-top: 4px solid #008CBA !important;
    box-shadow: 0 -3px 8px rgba(0,0,0,0.05);
}
/* =========================
   CARDS (KPIs + Panels)
========================= */
.kpi-card, .gr-box, .gr-panel {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.06) !important;
    color: #2D3748 !important;
    transition: all 0.25s ease-in-out;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.10) !important;
}
/* =========================
   INPUTS
========================= */
textarea, input, .gr-input {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E0 !important;
    color: #1a202c !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-size: 14px;
}
textarea:focus, input:focus {
    border-color: #008CBA !important;
    box-shadow: 0 0 0 3px rgba(0, 140, 186, 0.2) !important;
}
/* =========================
   LABELS
========================= */
label {
    color: #2C5282 !important;
    font-weight: 600;
    margin-bottom: 4px;
}
/* =========================
   PRIMARY BUTTON
========================= */
button.primary {
    background: linear-gradient(90deg, #008CBA 0%, #28B463 100%) !important;
    color: #ffffff !important;
    font-weight: bold;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 18px;
    transition: all 0.2s ease-in-out;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 140, 186, 0.35);
}
/* =========================
   STOP BUTTON
========================= */
button.stop-btn {
    background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%) !important;
    color: white !important;
    font-weight: bold;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 18px;
    margin-top: 15px !important;
}
button.stop-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(192, 57, 43, 0.35);
}
.gradio-container input[type="radio"]:checked + label {
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    color: white !important;
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: bold;
    box-shadow: 0 6px 18px rgba(14,165,233,0.4);
}
/* =========================
   FEEDBACK BUTTONS
========================= */
.feedback-btn-solved {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    font-weight: bold;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px;
    margin: 5px !important;
}
.feedback-btn-notsolved {
    background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%) !important;
    color: white !important;
    font-weight: bold;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px;
    margin: 5px !important;
}
/* =========================
   RATING STARS
========================= */
.rating-container {
    background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%) !important;
    border: 2px solid #fbbf24 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin: 15px 0 !important;
}
.star-rating {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin: 20px 0;
}
.star-btn {
    background: #e5e7eb !important;
    color: #9ca3af !important;
    border: 2px solid #d1d5db !important;
    border-radius: 50% !important;
    width: 60px !important;
    height: 60px !important;
    font-size: 24px !important;
    font-weight: bold !important;
    padding: 0 !important;
    transition: all 0.2s ease !important;
}
.star-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.star-btn.selected {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
    color: white !important;
    border-color: #f59e0b !important;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
}
.star-btn.selected i {
    color: white !important;
}
.star-btn.selected {
    transform: scale(1.15);
    transition: all 0.2s ease;
}
.no-rating-btn {
    background: #f3f4f6 !important;
    color: #6b7280 !important;
    border: 2px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    margin-top: 10px !important;
}
.no-rating-btn:hover {
    background: #e5e7eb !important;
    color: #4b5563 !important;
}
/* =========================
   TABLE
========================= */
.dataframe {
    background-color: white !important;
    color: #2D3748 !important;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
/* =========================
   EMPTY STATE FIX
========================= */
.gr-box:empty::before {
    content: "⏳ في انتظار إدخال البيانات...";
    color: #94a3b8;
    font-size: 14px;
    display: block;
    text-align: center;
    padding: 20px;
}
/* =========================
   PREDICTION CARD
========================= */
.prediction-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    border: 2px solid #0ea5e9 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin: 15px 0 !important;
}
/* =========================
   STATUS INDICATORS
========================= */
.status-solved {
    color: #10b981 !important;
    font-weight: bold;
}
.status-pending {
    color: #f59e0b !important;
    font-weight: bold;
}
.status-escalated {
    color: #ef4444 !important;
    font-weight: bold;
}
"""

# ==========================================
# 🤖 PREDICTION MODEL SETUP
# ==========================================
PRED_MODEL_PATH = "models/intelligent_behavioral_model.pkl"
ISSUE_ENCODER_PATH = "models/issue_encoder.pkl"
PATTERN_ENCODER_PATH = "models/pattern_encoder.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
SEQUENCE_PATH = "models/customer_sequences.pkl"

# Try to load models
try:
    pred_model = joblib.load(PRED_MODEL_PATH)
    issue_encoder = joblib.load(ISSUE_ENCODER_PATH)
    pattern_encoder = joblib.load(PATTERN_ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    sequence_df = joblib.load(SEQUENCE_PATH)
    print("✅ Prediction models loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading prediction assets: {e}")
    pred_model = None
    issue_encoder = None
    pattern_encoder = None
    feature_columns = None
    sequence_df = None

# ==========================================
# 🔐 ADMIN SETTINGS
# ==========================================
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "ADMIN_2026")
IS_TRAINING = False

# ==========================================
# 🤖 PART 1: AI AGENT LOGIC
# ==========================================
# Hugging Face Login
try:
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Logged into Hugging Face")
except Exception as e:
    print(f"⚠️ Hugging Face login error: {e}")

# Environment settings
os.environ["TRANSFORMERS_NO_TIKTOKEN"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

# Vector store setup
VECTOR_DIR = "vector_store"
FAISS_PATH = f"{VECTOR_DIR}/faiss.index"
META_PATH = f"{VECTOR_DIR}/meta.json"
os.makedirs(VECTOR_DIR, exist_ok=True)

SIMILARITY_THRESHOLD = 0.78

# SESSION STATE
SESSION_STATE = {
    "original_issue": None,
    "matches": [],
    "current_step": 0,
    "conversation": "",
    "last_solution_text": "",
    "case_id": None,
    "awaiting_feedback": False,
    "case_created": False,
    "resolution_steps_count": 0,
    "prediction_shown": False,
    "prediction_accepted": False,
    "predicted_issues": [],
    "customer_phone": None,
    "rating_submitted": False,
    "selected_rating": None,
    "feedback_submitted": False
}

# Load Whisper Model
try:
    whisper_model = WhisperModel(
        "large-v2",
        device="cuda" if device == "cuda" else "cpu",
        compute_type="float16" if device == "cuda" else "int8"
    )
    print("✅ Whisper model loaded")
except Exception as e:
    whisper_model = None
    print(f"⚠️ Error loading Whisper model: {e}")

TELECOM_CONTEXT_PROMPT = (
    "محادثة تقنية عن خدمات الإنترنت والاتصالات. "
    "مصطلحات: راوتر، VDSL، ADSL، باقة، Quota، جيجابايت، Landline، "
    "Splitter، Configuration، IP Address، Ping، Latency، فايبر."
)

def speech_to_text(audio_path):
    """Convert speech to text using Whisper"""
    if whisper_model is None:
        return ""
    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            language="ar",
            beam_size=4,
            initial_prompt=TELECOM_CONTEXT_PROMPT,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"Speech to text error: {e}")
        return ""

# Load LLM Model
try:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    llm = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    print("✅ LLM model loaded")
except Exception as e:
    tokenizer = None
    llm = None
    print(f"⚠️ Error loading LLM model: {e}")

FORMATTER_PROMPT = """
أنت منسق نصوص فقط.
مهمتك: تحويل "خطوات الحل الخام" المستخرجة من قاعدة البيانات إلى خطوات مرقمة واضحة للعميل.
القواعد الصارمة:
1. لا تضف أي معلومة من خارج النص.
2. لا تقترح حلولاً غير موجودة في الخطوات الخام.
3. التزم بالمصطلحات التقنية كما هي.
---
مثال:
المشكلة: النت بطيء
الخطوات الخام: ["رستر الراوتر", "تأكد من الحرارة"]
الرد:
1. قم بإعادة تشغيل الراوتر.
2. تأكد من وجود حرارة في خط الهاتف الأرضي.
---
البيانات الحالية:
المشكلة: {ISSUE}
الخطوات الخام:
{CONTEXT}
الرد المنسق (فقط):
"""

def llm_answer(context, issue):
    """Format troubleshooting steps using LLM"""
    if llm is None or tokenizer is None:
        # Fallback to simple formatting
        steps = context.split('\n')
        formatted = ""
        for i, step in enumerate(steps, 1):
            if step.strip():
                formatted += f"{i}. {step.strip()}\n"
        return formatted.strip()

    try:
        prompt = FORMATTER_PROMPT.format(ISSUE=issue, CONTEXT=context)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = llm.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.2
        )
        return tokenizer.decode(out[0], skip_special_tokens=True).split("الرد المنسق (فلا):")[-1].strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return context

# Load Knowledge Base
try:
    with open("data/telecom_data_egypt_20251225_cleaned.json", encoding="utf-8") as f:
        kb = json.load(f)
    print(f"✅ Knowledge base loaded with {len(kb)} entries")
except Exception as e:
    print(f"⚠️ Error loading knowledge base: {e}")
    kb = []

def normalize_text(text):
    """Normalize Arabic text for search"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text.strip()

# Prepare documents for FAISS
documents, meta = [], []
for row in kb:
    issue_txt = row.get('issue', '')
    cust_msg = row.get('customer_message', '')
    cat_txt = row.get('category', '')
    doc_content = f"{normalize_text(issue_txt)} {normalize_text(cust_msg)} {normalize_text(cat_txt)}"
    documents.append(doc_content)
    meta.append(row)

# Setup Embeddings and FAISS
try:
    embedder = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        try:
            index = faiss.read_index(FAISS_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            print("✅ FAISS index loaded from cache")
        except Exception as e:
            print(f"⚠️ Error loading cached FAISS index, rebuilding: {e}")
            embeddings = embedder.encode(documents, normalize_embeddings=True, convert_to_numpy=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, FAISS_PATH)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
    else:
        embeddings = embedder.encode(documents, normalize_embeddings=True, convert_to_numpy=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print("✅ FAISS index created and saved")
except Exception as e:
    print(f"⚠️ FAISS error: {e}")
    index = None
    embedder = None

def retrieve(issue, k=3):
    """Retrieve similar cases from FAISS index"""
    if index is None or embedder is None:
        return [], 0.0

    clean_q = normalize_text(issue)
    q_emb = embedder.encode([f"query: {clean_q}"], normalize_embeddings=True)
    scores, ids = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if score >= SIMILARITY_THRESHOLD:
            results.append({"data": meta[idx], "score": float(score)})
    if not results:
        return [], 0.0
    top_score = results[0]["score"]
    return results, top_score

# ==========================================
# 📞 CRM DATA LOADING
# ==========================================
def normalize_phone(phone):
    """
    Normalize Egyptian phone numbers to unified format: 01XXXXXXXXX
    Handles Excel / Google Sheets numeric issues safely.
    """
    if phone is None:
        return None

    phone = str(phone).strip()

    # Remove trailing .0 from floats
    if phone.endswith(".0"):
        phone = phone[:-2]

    # Keep digits only
    phone = re.sub(r"\D", "", phone)

    # Handle country code
    if phone.startswith("0020"):
        phone = phone[4:]
    elif phone.startswith("20"):
        phone = phone[2:]

    # Handle missing leading zero
    if len(phone) == 10 and phone.startswith("1"):
        phone = "0" + phone

    # Final validation for Egyptian mobiles
    if len(phone) == 11 and phone.startswith(("010", "011", "012", "015")):
        return phone

    return None


# Load CRM data
try:
    crm = pd.read_excel("data/customer_subscriptions_2000_final.xlsx")
    crm["phone_number"] = crm["phone_number"].apply(normalize_phone)
    crm = crm.dropna(subset=["phone_number"])
    print(f"✅ CRM data loaded with {len(crm)} customers")
except Exception as e:
    crm = pd.DataFrame()
    print(f"⚠️ Error loading CRM data: {e}")

def get_customer(phone):
    """Get customer information from CRM"""
    if crm.empty:
        return None
    
    phone = normalize_phone(phone)
    if not phone:
        return None
    
    res = crm[crm["phone_number"] == phone]
    return None if res.empty else res.iloc[0].to_dict()

# ==========================================
# 📄 GOOGLE SHEET EXPECTED HEADERS
# ==========================================
EXPECTED_HEADERS = [
    "case_id",
    "created_at",
    "resolved_at",
    "customer_name",
    "phone_number",
    "subscription_type",
    "bundle_price",
    "issue_text",
    "solution_status",
    "resolution_steps_count",
    "resolved_by_step",
    "confidence",
    "rating",
    "prediction_shown",
    "prediction_accepted",
    "issue_category",
    "category_confidence",
    "category_source"
]

ISSUE_TRANSLATIONS = {
    "internet_down": "انقطاع الإنترنت",
    "slow_internet": "بطء في سرعة الإنترنت",
    "router_issue": "مشكلة في الراوتر",
    "offers_inquiry": "الاستفسار عن العروض",
    "billing_issue": "مشكلة في الفاتورة",
    "international_roaming": "مشكلة خدمة التجوال الدولي",
    "landline_down": "مشكلة خط أرضي",
    "other": "مشكلة أخرى"
}

def translate_issue(issue):
    """Translate issue category to Arabic"""
    return ISSUE_TRANSLATIONS.get(issue, issue)

# ==========================================
# 🕒 TIMEZONE (Egypt)
# ==========================================
EGYPT_TZ = pytz.timezone("Africa/Cairo")

def now_egypt():
    """Get current time in Egypt timezone"""
    return datetime.now(EGYPT_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==========================================
# 🔍 ISSUE CLASSIFICATION
# ==========================================
def classify_issue(text):
    """Classify issue text into categories"""
    text_norm = normalize_text(text)
    text_tokens = set(text_norm.split())
    
    intent_scores = {}
    
    INTENTS = {
        "INT_INTERNET_DOWN": {
            "patterns": [
                "انترنت فصل",
                "انترنت مش شغال",
                "الخدمه فاصله",
                "dsl فصل",
                "الانترنت فاصل",
                "internet down",
                "no internet"
            ],
            "category": "internet_down",
            "weight": 1.0
        },
        "INT_SLOW_INTERNET": {
            "patterns": [
                "انترنت بطيء",
                "السرعه ضعيفه",
                "slow internet",
                "ping عالي",
                "lag",
                "النت مش ثابت"
            ],
            "category": "slow_internet",
            "weight": 0.9
        },
        "INT_ROUTER_ISSUE": {
            "patterns": [
                "راوتر",
                "لمبه حمرا",
                "wlan",
                "reset",
                "اعدادات طارت"
            ],
            "category": "router_issue",
            "weight": 0.9
        },
        "INT_LANDLINE_DOWN": {
            "patterns": [
                "خط ارضي",
                "حراره",
                "سماعه",
                "التليفون الارضي فاصل"
            ],
            "category": "landline_down",
            "weight": 1.0
        },
        "INT_BILLING": {
            "patterns": [
                "فاتوره",
                "مبلغ",
                "خصم",
                "زياده",
                "محاسبه",
                "payment",
                "charge"
            ],
            "category": "billing_issue",
            "weight": 1.0
        },
        "INT_OFFERS": {
            "patterns": [
                "عروض",
                "باقات",
                "ترقيه",
                "نظام",
                "اشتراك",
                "vip"
            ],
            "category": "offers_inquiry",
            "weight": 0.8
        },
        "INT_ROAMING": {
            "patterns": [
                "سعوديه",
                "دبي",
                "تجوال",
                "roaming",
                "مسافر"
            ],
            "category": "international_roaming",
            "weight": 1.0
        }
    }
    
    CONTEXT_BOOST = {
        "internet_down": {
            "signals": ["فصل", "فاصل", "مش شغال", "واقف", "من يوم", "ساعه", "يومين"],
            "boost": 0.4
        },
        "slow_internet": {
            "signals": ["بطيء", "ضعيف", "مش ثابت", "lag", "ping", "2 ميجا"],
            "boost": 0.35
        },
        "router_issue": {
            "signals": ["لمبه", "حمرا", "راوتر", "reset", "wlan", "dsl"],
            "boost": 0.35
        },
        "billing_issue": {
            "signals": ["فاتوره", "دافع", "مبلغ", "خصم", "زياده", "80%"],
            "boost": 0.45
        },
        "offers_inquiry": {
            "signals": ["باقه", "عرض", "ترقيه", "vip", "اشترك"],
            "boost": 0.3
        },
        "international_roaming": {
            "signals": ["سعوديه", "دبي", "مسافر", "تجوال"],
            "boost": 0.5
        },
        "landline_down": {
            "signals": ["حراره", "خط ارضي", "سماعه", "تليفون"],
            "boost": 0.5
        }
    }
    
    # 1️⃣ Intent Matching (Token Overlap)
    for intent_name, intent in INTENTS.items():
        score = 0.0
        
        for pattern in intent["patterns"]:
            pattern_clean = normalize_text(pattern)
            pattern_tokens = set(pattern_clean.split())
            
            if not pattern_tokens:
                continue
            
            overlap = len(text_tokens & pattern_tokens)
            match_ratio = overlap / len(pattern_tokens)
            
            if match_ratio >= 0.5:
                score += intent["weight"] * match_ratio
        
        if score > 0:
            intent_scores[intent_name] = score
    
    # 2️⃣ Contextual Boost Layer
    for category, cfg in CONTEXT_BOOST.items():
        if any(sig in text_norm for sig in cfg["signals"]):
            for intent_name, intent in INTENTS.items():
                if intent["category"] == category:
                    intent_scores[intent_name] = (
                        intent_scores.get(intent_name, 0) + cfg["boost"]
                    )
    
    # 3️⃣ Short Text Fallback
    if not intent_scores and len(text_tokens) <= 3:
        if any(w in text_norm for w in ["فصل", "فاصل", "واقف"]):
            return "internet_down", 0.55, "context_short"
        if any(w in text_norm for w in ["بطيء", "ضعيف", "lag", "ping"]):
            return "slow_internet", 0.5, "context_short"
    
    # 4️⃣ No Match
    if not intent_scores:
        return "other", 0.0, "unclassified"
    
    # 5️⃣ Pick Best Intent
    best_intent = max(intent_scores, key=intent_scores.get)
    intent_cfg = INTENTS[best_intent]
    confidence = min(intent_scores[best_intent], 1.0)
    
    return (
        intent_cfg["category"],
        round(confidence, 2),
        "intent"
    )

# ==========================================
# 📊 GOOGLE SHEET OPERATIONS
# ==========================================
def create_case(customer, phone, issue_text, confidence):
    """Create a new case in Google Sheets (SAFE for unregistered customers)"""

    # ✅ SAFETY FIX: handle unregistered customer
    if not customer or not isinstance(customer, dict):
        customer = {
            "customer_name": "غير مسجل",
            "subscription_type": "غير مشترك",
            "bundle_price": "غير متاح"
        }

    case_id = str(uuid.uuid4())
    cat, cat_conf, cat_src = classify_issue(issue_text)

    row = [
        case_id,
        now_egypt(),
        "",
        customer.get("customer_name"),
        phone,
        customer.get("subscription_type"),
        customer.get("bundle_price"),
        issue_text,
        "قيد المعالجة",
        1,
        "",
        confidence,
        "",
        SESSION_STATE["prediction_shown"],
        SESSION_STATE["prediction_accepted"],
        cat,
        cat_conf,
        cat_src
    ]

    if sheet:
        try:
            sheet.append_row(row)
            print(f"✅ Case created: {case_id}")
        except Exception as e:
            print(f"⚠️ Error creating case in Google Sheets: {e}")

    return case_id


def update_case_steps(case_id, steps_count):
    """Update resolution steps count in Google Sheets"""
    if sheet is None:
        return
    
    try:
        records = sheet.get_all_records()
        for i, r in enumerate(records, start=2):
            if r.get("case_id") == case_id:
                sheet.update_cell(i, 10, steps_count)  # resolution_steps_count
                break
    except Exception as e:
        print(f"Update case steps error: {e}")

def close_case(case_id, rating=None, resolved_by_step=""):
    """Close a case in Google Sheets"""
    if sheet is None:
        return
    
    resolved_at = now_egypt()
    try:
        records = sheet.get_all_records()
        for i, r in enumerate(records, start=2):
            if r.get("case_id") == case_id:
                sheet.update_cell(i, 3, resolved_at)      # resolved_at
                sheet.update_cell(i, 9, "اتحلت")          # solution_status
                sheet.update_cell(i, 11, resolved_by_step)
                if rating:
                    sheet.update_cell(i, 13, rating)
                break
    except Exception as e:
        print(f"Close case error: {e}")

# ==========================================
# 📈 DATA LOADING FOR DASHBOARD
# ==========================================
def load_data():
    """Load data from Google Sheets"""
    try:
        if sheet:
            records = sheet.get_all_records()
            df = pd.DataFrame(records)
        else:
            # Fallback to CSV if Google Sheets not available
            GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1GJPnnylIcCymLBBfBtR0QeQAvXa91OOaID7U3-0sqXg/export?format=csv"
            df = pd.read_csv(GSHEET_CSV_URL)
        # ======================================
        # 🔧 FIX BOOLEAN COLUMNS (VERY IMPORTANT)

        def fix_bool(x):
            return str(x).strip().lower() in ["true", "1", "yes"]
    

        for col in ["prediction_shown", "prediction_accepted"]:
            if col in df.columns:
                df[col] = df[col].apply(fix_bool)
        
            else:
                df[col] = False
        
    

        # Data processing
        if 'created_at' in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        if 'resolved_at' in df.columns:
            df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce")
        
        if 'solution_status' in df.columns:
            df["solution_status"] = df["solution_status"].fillna("قيد المعالجة")
        
        if 'rating' in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        
        # Calculate resolution time
        if all(col in df.columns for col in ['resolved_at', 'created_at']):
            df["resolution_minutes"] = ((df["resolved_at"] - df["created_at"]).dt.total_seconds() / 60)
        
        # Calculate weekday
        if 'created_at' in df.columns:
            df["weekday"] = df["created_at"].dt.day_name()
        
        # Handle prediction columns
        if 'prediction_shown' not in df.columns:
            df["prediction_shown"] = False
        if 'prediction_accepted' not in df.columns:
            df["prediction_accepted"] = False
        
        df["prediction_shown"] = df["prediction_shown"].fillna(False)
        df["prediction_accepted"] = df["prediction_accepted"].fillna(False)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# ==========================================
# 🤖 PREDICTION FUNCTIONS
# ==========================================
def extract_customer_sequences(df):
    """Extract customer sequences from data"""
    if df.empty:
        return pd.DataFrame()
    
    sequences = []
    for phone, group in df.groupby("phone_number"):
        if len(group) >= 2:
            group = group.sort_values("created_at")
            issues = group["issue_category"].tolist()
            for i in range(len(issues) - 1):
                sequences.append({
                    "phone_number": phone,
                    "current_issue": issues[i],
                    "next_issue": issues[i + 1]
                })
    
    return pd.DataFrame(sequences)

def create_pattern_features(sequence_df, history_df):
    """Create pattern features for prediction"""
    if sequence_df is None or sequence_df.empty or history_df.empty:
        return {
            "pattern_match_count": 0,
            "pattern_confidence": 0,
            "current_issue_frequency": 0,
            "issue_variety": 0,
            "customer_most_common_next": None
        }
    
    current_issue = history_df.iloc[-1]["issue_category"] if not history_df.empty else None
    
    # ✅ تحقق من وجود الأعمدة المطلوبة
    if "phone_number" not in sequence_df.columns or "current_issue" not in sequence_df.columns:
        return {
            "pattern_match_count": 0,
            "pattern_confidence": 0,
            "current_issue_frequency": 0,
            "issue_variety": 0,
            "customer_most_common_next": None
        }
    
    phone_number = history_df.iloc[0]["phone_number"] if not history_df.empty else None
    
    if not phone_number or not current_issue:
        return {
            "pattern_match_count": 0,
            "pattern_confidence": 0,
            "current_issue_frequency": 0,
            "issue_variety": 0,
            "customer_most_common_next": None
        }
    
    # Calculate pattern matches
    pattern_matches = sequence_df[
        (sequence_df["phone_number"] == phone_number) &
        (sequence_df["current_issue"] == current_issue)
    ]
    
    if not pattern_matches.empty:
        most_common = pattern_matches["next_issue"].mode()
        most_common_next = most_common.iloc[0] if not most_common.empty else None
        pattern_confidence = len(pattern_matches) / max(1, len(sequence_df))
    else:
        most_common_next = None
        pattern_confidence = 0
    
    return {
        "pattern_match_count": len(pattern_matches),
        "pattern_confidence": pattern_confidence,
        "current_issue_frequency": (history_df["issue_category"] == current_issue).sum(),
        "issue_variety": history_df["issue_category"].nunique(),
        "customer_most_common_next": most_common_next
    }


def predict_next_issue_intelligent(phone, df_hist, top_k=3):
    """Intelligent prediction of next issue"""
    phone = normalize_phone(phone)
    if not phone or pred_model is None or sequence_df is None:
        return {"success": False, "predictions": []}
    
    customer_history = df_hist[df_hist["phone_number"] == phone]
    if len(customer_history) < 2:
        return {"success": False, "predictions": []}
    
    customer_history = customer_history.sort_values("created_at").reset_index(drop=True)
    latest_issue = customer_history.iloc[-1]["issue_category"]
    
    # Prepare features
    pattern_features = create_pattern_features(sequence_df, customer_history)
    
    # التحقق من وجود issue_encoder
    if issue_encoder is None:
        return {"success": False, "predictions": []}
    
    # Encode current issue
    try:
        current_issue_enc = issue_encoder.transform([latest_issue])[0]
    except:
        return {"success": False, "predictions": []}
    
    # التحقق من وجود pattern_encoder
    if pattern_encoder is None:
        return {"success": False, "predictions": []}
    
    # Prepare feature vector
    features = {
        "current_issue_enc": current_issue_enc,
        "resolution_minutes": customer_history.iloc[-1].get("resolution_minutes", 30),
        "case_index": len(customer_history),
        "days_since_last": (
            (customer_history.iloc[-1]["created_at"] - customer_history.iloc[-2]["created_at"]).days
            if len(customer_history) > 1 else 0
        ),
        "pattern_match_count": pattern_features["pattern_match_count"],
        "pattern_confidence": pattern_features["pattern_confidence"],
        "current_issue_frequency": pattern_features["current_issue_frequency"],
        "issue_variety": pattern_features["issue_variety"],
        "customer_pattern_enc": pattern_encoder.transform([pattern_features.get("customer_most_common_next", "unknown")])[0]
    }
    
    # التحقق من وجود feature_columns
    if feature_columns is None:
        return {"success": False, "predictions": []}
    
    # Create feature array
    feature_array = np.array([[features[col] for col in feature_columns]])
    
    # Predict
    try:
        predictions = pred_model.predict_proba(feature_array)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        
        result = []
        for idx in top_indices:
            if predictions[idx] > 0.1:  # Minimum confidence threshold
                issue = issue_encoder.inverse_transform([idx])[0]
                result.append({
                    "issue": issue,
                    "confidence": float(predictions[idx])
                })
        
        return {"success": True, "predictions": result}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"success": False, "predictions": []}

def predict_next_issue(phone, top_k=3):
    """Predict next issue for a customer"""
    phone = normalize_phone(phone)
    if not phone or pred_model is None:
        return []
    
    df_hist = load_data()
    df_hist["phone_number"] = df_hist["phone_number"].apply(normalize_phone)
    
    customer_history = df_hist[df_hist["phone_number"] == phone]
    if len(customer_history) < 2:
        return []
    
    result = predict_next_issue_intelligent(phone, df_hist, top_k)
    
    if not result.get("success"):
        return []
    
    return [
        translate_issue(p["issue"])
        for p in result["predictions"]
        if p["confidence"] >= 0.25
    ]

# ==========================================
# 🔁 SELF-LEARNING FUNCTIONS
# ==========================================
def is_high_quality_sample(row):
    """Check if a sample is high quality for training"""
    return (
        row.get("prediction_accepted") is True and
        row.get("rating") is not None and
        row.get("rating") >= 4 and
        row.get("issue_category") not in ["other", None]
    )

def build_training_dataset(df):
    """Build training dataset from high quality samples"""
    high_quality = df[df.apply(is_high_quality_sample, axis=1)]
    
    if len(high_quality) < 50:
        return None
    
    return high_quality[
        ["phone_number", "created_at", "resolved_at", "issue_category"]
    ].dropna()

def should_retrain(df, min_cases=50):
    """Check if retraining should occur"""
    labeled = df[df["issue_category"].notna()]
    return len(labeled) >= min_cases

def retrain_behavioral_model():
    """Retrain the behavioral model"""
    global IS_TRAINING
    global pred_model, issue_encoder, pattern_encoder, feature_columns, sequence_df
    
    print("🔁 Starting retraining process...")
    IS_TRAINING = True
    
    try:
        # Load data
        df = load_data()
        if df.empty:
            return "❌ No data available"
        
        # Check retraining condition
        if not should_retrain(df):
            return "❌ Not enough labeled data"
        
        high_quality_df = build_training_dataset(df)
        if high_quality_df is None:
            return "❌ Not enough high-quality samples"
        
        # Extract sequences
        sequence_df = extract_customer_sequences(high_quality_df)
        
        # Build training dataset
        train_data = []
        
        for phone, group in high_quality_df.groupby("phone_number"):
            group = group.sort_values("created_at").reset_index(drop=True)
            
            if len(group) < 3:
                continue
            
            for i in range(1, len(group) - 1):
                history = group.iloc[:i+1]
                
                base_features = {
                    "current_issue": group.iloc[i]["issue_category"],
                    "next_issue": group.iloc[i+1]["issue_category"],
                    "resolution_minutes": (
                        (group.iloc[i]["resolved_at"] - group.iloc[i]["created_at"]).total_seconds() / 60
                        if pd.notna(group.iloc[i]["resolved_at"])
                        else 30
                    ),
                    "case_index": i,
                    "days_since_last": (
                        (group.iloc[i]["created_at"] - group.iloc[i-1]["created_at"]).days
                        if i > 0 else 0
                    )
                }
                
                pattern_features = create_pattern_features(sequence_df, history)
                train_data.append({**base_features, **pattern_features})
        
        train_df = pd.DataFrame(train_data)
        if train_df.empty:
            return "❌ Training dataframe empty"
        
        # Encoding
        issue_encoder = LabelEncoder()
        issue_encoder.fit(
            pd.concat([train_df["current_issue"], train_df["next_issue"]]).unique()
        )
        
        train_df["current_issue_enc"] = issue_encoder.transform(train_df["current_issue"])
        train_df["next_issue_enc"] = issue_encoder.transform(train_df["next_issue"])
        
        pattern_encoder = LabelEncoder()
        pattern_encoder.fit(train_df["customer_most_common_next"].fillna("unknown"))
        train_df["customer_pattern_enc"] = pattern_encoder.transform(
            train_df["customer_most_common_next"].fillna("unknown")
        )
        
        feature_columns = [
            "current_issue_enc",
            "resolution_minutes",
            "case_index",
            "days_since_last",
            "pattern_match_count",
            "pattern_confidence",
            "current_issue_frequency",
            "issue_variety",
            "customer_pattern_enc"
        ]
        
        X = train_df[feature_columns]
        y = train_df["next_issue_enc"]
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model training
        models = [
            RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),
            XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, eval_metric="mlogloss", random_state=42),
            LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, class_weight="balanced", random_state=42)
        ]
        
        best_model, best_score = None, 0
        
        for model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = recall_score(y_test, preds, average="macro", zero_division=0)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Save artifacts
        joblib.dump(best_model, PRED_MODEL_PATH)
        joblib.dump(issue_encoder, ISSUE_ENCODER_PATH)
        joblib.dump(pattern_encoder, PATTERN_ENCODER_PATH)
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
        joblib.dump(sequence_df, SEQUENCE_PATH)
        
        # Update global variables
        
        pred_model = best_model
        issue_encoder = issue_encoder
        pattern_encoder = pattern_encoder
        feature_columns = feature_columns
        sequence_df = sequence_df
        
        print("✅ Retraining finished successfully")
        return "✅ Model retrained successfully"
        
    except Exception as e:
        print(f"❌ Retraining error: {e}")
        return f"❌ Error: {str(e)}"
    finally:
        IS_TRAINING = False

# ==========================================
# 🎛️ UI HELPER FUNCTIONS
# ==========================================
def confirm_customer(name, phone):
    """Confirm customer and show predictions safely (Gradio-safe)"""
    phone = normalize_phone(phone)

    # 🔒 Reset session prediction-related state
    SESSION_STATE.update({
        "prediction_shown": False,
        "prediction_accepted": False,
        "predicted_issues": [],
        "case_id": None,
        "customer_phone": phone,
        "rating_submitted": False,
        "feedback_submitted": False
    })

    # ❌ Invalid phone
    if not phone:
        return (
            gr.update(visible=False),                 # pred_radio
            "⚠️ رقم الهاتف غير صالح",                 # out_response
            gr.update(visible=False),                 # feedback_container
            ""                                        # out_crm
        )

    # 📞 Get customer data
    customer = get_customer(phone)

    if customer:
        crm_text = (
            f"👤 العميل: {customer.get('customer_name', 'غير معروف')}\n"
            f"📱 الهاتف: {phone}\n"
            f"📦 الباقة: {customer.get('subscription_type', 'غير متاح')}\n"
            f"💰 السعر: {customer.get('bundle_price', 'غير متاح')}"
        )
    else:
        crm_text = (
            f"👤 العميل: غير مسجل\n"
            f"📱 الهاتف: {phone}\n"
            f"📦 الحالة: غير مشترك"
        )

    # 🔮 Predict next issue
    try:
        predicted = predict_next_issue(phone)
    except Exception as e:
        print(f"Prediction error in confirm_customer: {e}")
        predicted = []

    if predicted:
        SESSION_STATE["prediction_shown"] = True
        SESSION_STATE["predicted_issues"] = predicted

        return (
            gr.update(
                choices=predicted + ["مشكلة أخرى"],
                visible=True,
                value=None,
                label="🔮 بناءً على سجلك السابق، اختر المشكلة المتوقعة:"
            ),
            f"✅ تم تأكيد بيانات العميل.\n🔮 تم العثور على {len(predicted)} مشكلة متوقعة",
            gr.update(visible=False),
            crm_text
        )

    # ℹ️ No predictions
    return (
        gr.update(visible=False),
        "✅ تم تأكيد بيانات العميل.\nℹ️ لا توجد تنبؤات متاحة",
        gr.update(visible=False),
        crm_text
    )
def pipeline(name, phone, text_issue, audio, pred_radio):
    """Main pipeline for issue processing"""
    phone = normalize_phone(phone)
    
    if not phone:
        return (
        "",                     # out_issue
        "",                     # out_crm
        "⚠️ رقم الهاتف غير صالح",  # out_response
        gr.update(visible=False),  # feedback_container
        gr.update(visible=False),  # rating_container
        ""                      # rating_value
    )
    
    customer = get_customer(phone)
    crm_text = (
        f"👤 العميل: {customer.get('customer_name','غير معروف')}\n"
        f"📱 الهاتف: {phone}\n"
        f"📦 الباقة: {customer.get('subscription_type','غير متاح')}"
    ) if customer else f"👤 العميل: غير مسجل\n📱 الهاتف: {phone}"
    

    
    # Determine the issue
    issue = None
    # 1️⃣ Priority to written text
    if text_issue and text_issue.strip():
        issue = text_issue.strip()
    # 2️⃣ Then audio
    elif audio:
        issue = speech_to_text(audio)
    # 3️⃣ Then prediction (optional)
    elif pred_radio and pred_radio != "مشكلة أخرى":
        issue = pred_radio
        SESSION_STATE["prediction_accepted"] = True
    # 4️⃣ No input
    else:
        return (
            "",
            crm_text,
            "✍️ من فضلك اكتب المشكلة أو استخدم التسجيل الصوتي (التنبؤ اختياري)",
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )
    
    # Retrieval
    matches, score = retrieve(issue)
    if not matches:
        return issue or "",crm_text or "", "⚠️ لا توجد حلول مسجلة", gr.update(visible=False), gr.update(visible=False), ""
    
    # Create case
    case_id = create_case(customer, phone, issue, round(score * 100))
    
    SESSION_STATE.update({
        "case_id": case_id,
        "original_issue": issue,
        "matches": matches,
        "current_step": 0,
        "resolution_steps_count": 1
    })
    
    steps = matches[0]["data"].get("troubleshooting_steps", [])
    answer = llm_answer("\n".join(steps), issue)
    
    SESSION_STATE["last_solution_text"] = answer
    
    msg = (
        f"🆔 رقم الحالة: {case_id}\n"
        f"🛠 الحل المقترح:\n{answer}\n\n"
        f"❓ هل تم حل المشكلة؟"
    )
    
    return issue or "", crm_text or "", msg, gr.update(visible=True), gr.update(visible=False), ""


def handle_solved():
    """Handle solved issue"""
    case_id = SESSION_STATE.get("case_id")
    
    if SESSION_STATE.get("feedback_submitted"):
        return (
            "ℹ️ تم تسجيل حل المشكلة بالفعل.",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    resolved_steps = SESSION_STATE.get("last_solution_text", "")
    
    if sheet and case_id:
        try:
            records = sheet.get_all_records()
            for i, r in enumerate(records, start=2):
                if r.get("case_id") == case_id:
                    sheet.update_cell(i, 3, now_egypt())     # resolved_at
                    sheet.update_cell(i, 9, "اتحلت")         # solution_status
                    sheet.update_cell(i, 11, resolved_steps) # resolved_by_step
                    break
        except Exception as e:
            print(f"Error closing case: {e}")
    
    SESSION_STATE.update({
        "feedback_submitted": True,
        "rating_submitted": False
    })
    
    return (
        "✅ تم حل المشكلة بنجاح.\n\n⭐ هل تريد تقييم الخدمة؟ (اختياري)",
        gr.update(visible=False),
        gr.update(visible=True)
    )

def handle_not_solved():
    """Handle not solved issue"""
    if SESSION_STATE.get("feedback_submitted"):
        return (
            "ℹ️ تم تسجيل رد بالفعل.",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    case_id = SESSION_STATE.get("case_id")
    matches = SESSION_STATE.get("matches", [])
    
    step = SESSION_STATE["current_step"] + 1
    
    # Escalate after exhausting all solutions
    if step >= len(matches):
        if sheet and case_id:
            try:
                records = sheet.get_all_records()
                for i, r in enumerate(records, start=2):
                    if r.get("case_id") == case_id:
                        sheet.update_cell(i, 9, "قيد المعالجة")
                        break
            except Exception as e:
                print(f"Error updating case status: {e}")
        
        return (
            "🚨 لم تنجح الحلول المتاحة.\n📨 تم تصعيد الحالة للدعم الفني.",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    # Alternative solution
    SESSION_STATE["current_step"] = step
    SESSION_STATE["resolution_steps_count"] += 1
    
    update_case_steps(case_id, SESSION_STATE["resolution_steps_count"])
    
    steps = matches[step]["data"].get("troubleshooting_steps", [])
    answer = llm_answer("\n".join(steps), SESSION_STATE["original_issue"])
    
    SESSION_STATE["last_solution_text"] = answer
    
    msg = (
        f"🔁 محاولة رقم {step + 1}:\n{answer}\n\n"
        f"❓ هل المشكلة اتحلت؟"
    )
    
    return msg, gr.update(visible=True), gr.update(visible=False)

def submit_rating(rating):
    """Submit rating"""
    case_id = SESSION_STATE.get("case_id")
    
    if not case_id:
        return "❌ لا توجد حالة نشطة للتقييم", gr.update(visible=False)
    
    if not rating:
        return "⭐ من فضلك اختر عدد النجوم أولاً", gr.update(visible=True)
    
    if sheet:
        try:
            records = sheet.get_all_records()
            for i, r in enumerate(records, start=2):
                if r.get("case_id") == case_id:
                    sheet.update_cell(i, 13, rating)
                    break
        except Exception as e:
            print(f"Error submitting rating: {e}")
    
    SESSION_STATE["rating_submitted"] = True
    return f"🙏 شكرًا لتقييمك ({rating}/5)", gr.update(visible=False)

def skip_rating():
    """Skip rating"""
    SESSION_STATE["rating_submitted"] = True
    skip_msg = "✅ تم إغلاق الحالة.\n\n💡 يمكنك بدء حالة جديدة."
    return skip_msg, gr.update(visible=False)


# ==========================================
# 📊 DASHBOARD FUNCTIONS
# ==========================================
WEEK_ORDER = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def get_base64_image(image_path):
    """Get base64 encoded image"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return ""

# Try to load logo - VERSION FIXED FOR HUGGING FACE
image_base64 = ""
try:
    # Try multiple possible locations
    possible_paths = [
        "/app/logo.png",        # Docker container path
        "logo.png",             # Current directory
        "assets/logo.png",      # Assets folder
        "./logo.png",           # Relative path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found logo at: {path}")
            image_base64 = get_base64_image(path)
            if image_base64:
                break
except Exception as e:
    print(f"⚠️ Logo loading error: {e}")

# Fallback: Use direct Hugging Face URL if local file not found
if not image_base64:
    try:
        # Direct URL to your logo on Hugging Face
        LOGO_URL = "https://huggingface.co/spaces/youssefmoustafa172/VoiceCare-AI/resolve/main/logo.png"
        import requests
        response = requests.get(LOGO_URL, timeout=5)
        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            print("✅ Loaded logo from Hugging Face URL")
    except:
        print("⚠️ Could not load logo from any source")

LOGO_HTML_SRC = f"data:image/png;base64,{image_base64}" if image_base64 else ""

def compute_kpis(df):
    """Compute KPI metrics"""
    total = int(len(df))
    
    solved = int((df["solution_status"] == "اتحلت").sum()) if "solution_status" in df.columns else 0
    solve_rate = round((solved / total) * 100, 1) if total else 0
    
    avg_rating = float(df["rating"].dropna().mean()) if "rating" in df.columns and df["rating"].notna().any() else 0.0
    avg_time = float(df["resolution_minutes"].dropna().mean()) if "resolution_minutes" in df.columns and df["resolution_minutes"].notna().any() else 0.0
    
    return (
        total,
        f"{solve_rate}%",
        round(avg_rating, 2),
        f"{round(avg_time,1)} min"
    )

def compute_prediction_kpis(df):
    """Compute prediction KPIs"""
    if 'prediction_shown' not in df.columns or 'prediction_accepted' not in df.columns:
        return 0, 0, "0%"
    
    shown = df[df["prediction_shown"] == True]
    accepted = df[df["prediction_accepted"] == True]
    
    rate = round((len(accepted) / len(shown)) * 100, 1) if len(shown) else 0
    return len(shown), len(accepted), f"{rate}%"

def plot_trend_weekday(df):
    """Plot weekly trend"""
    if df.empty or 'weekday' not in df.columns:
        fig = px.line(title="لا توجد بيانات", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    trend = df.groupby("weekday")["case_id"].count().reindex(WEEK_ORDER, fill_value=0).reset_index()
    fig = px.line(trend, x="weekday", y="case_id", markers=True, title="الاتجاه الأسبوعي", template="plotly_dark")
    fig.update_traces(line_color='#4ecca3', line_width=4, marker=dict(size=10, color='#ffffff', line=dict(width=2, color='#4ecca3')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#a0a0c0")
    return fig

def plot_status_pie(df):
    """Plot status distribution"""
    if df.empty or 'solution_status' not in df.columns:
        fig = px.pie(title="لا توجد بيانات", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        return fig
    
    pie_df = df["solution_status"].value_counts().reset_index()
    fig = px.pie(pie_df, names="solution_status", values="count", hole=0.5, title="توزيع الحالات", template="plotly_dark",
                 color_discrete_sequence=['#4ecca3', '#45b7d1', '#706fd3'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#a0a0c0")
    return fig

def plot_rating_histogram(df):
    """Plot rating histogram"""
    if df.empty or 'rating' not in df.columns:
        fig = px.bar(title="لا توجد بيانات", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    dff = df[df["rating"].between(1, 5)]
    if dff.empty:
        fig = px.bar(title="No Rating Data", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    fig = px.histogram(dff, x="rating", nbins=5, title="تحليل التقييمات", template="plotly_dark", color_discrete_sequence=['#45b7d1'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', bargap=0.2, font_color="#a0a0c0")
    return fig

def dashboard_view(sub, status, year, month):
    """Generate dashboard view"""
    try:
        df = load_data()
        if df.empty:
            empty_fig = px.scatter(title="لا توجد بيانات")
            empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return 0, "0%", 0, "0 min", 0, "0%", empty_fig, empty_fig, empty_fig, pd.DataFrame()
        
        dff = df.copy()
        
        # Apply filters
        if sub != "All" and 'subscription_type' in dff.columns:
            dff = dff[dff["subscription_type"] == sub]
        if status != "All" and 'solution_status' in dff.columns:
            dff = dff[dff["solution_status"] == status]
        if year != "All" and 'created_at' in dff.columns:
            dff = dff[dff["created_at"].dt.year == int(year)]
        if month != "All" and 'created_at' in dff.columns:
            dff = dff[dff["created_at"].dt.month == int(month)]
        
        if dff.empty:
            empty_fig = px.scatter(title="لا توجد بيانات")
            empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return 0, "0%", 0, "0 min", 0, "0%", empty_fig, empty_fig, empty_fig, pd.DataFrame()
        
        # Compute KPIs
        total, rate, avg_rating, avg_time = compute_kpis(dff)
        
        # Prediction KPIs
        pred_shown, pred_accepted, pred_rate = compute_prediction_kpis(dff)
        
        # Create charts
        fig1 = plot_trend_weekday(dff)
        fig2 = plot_status_pie(dff)
        fig3 = plot_rating_histogram(dff)
        
        # Prepare table for display
        display_df = dff.copy()
        if 'created_at' in display_df.columns:
            display_df = display_df.sort_values("created_at", ascending=False)
            display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
        
        if 'resolved_at' in display_df.columns:
            display_df["resolved_at"] = display_df["resolved_at"].dt.strftime("%Y-%m-%d %H:%M")
        
        # Select important columns
        important_cols = []
        for col in ['case_id', 'created_at', 'customer_name', 'phone_number',
                   'issue_text', 'solution_status', 'rating']:
            if col in display_df.columns:
                important_cols.append(col)
        
        if important_cols:
            display_df = display_df[important_cols]
        
        return (
            total,
            rate,
            avg_rating,
            avg_time,
            pred_shown,
            pred_rate,
            fig1,
            fig2,
            fig3,
            display_df
        )
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        empty_fig = px.scatter(title="خطأ في تحميل البيانات")
        empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return 0, "0%", 0, "0 min", 0, "0%", empty_fig, empty_fig, empty_fig, pd.DataFrame()

# ==========================================
# 🆕 FULL RESET & REFRESH FUNCTION
# ==========================================
def reset_session():
    """Reset session state"""
    SESSION_STATE.update({
        "original_issue": None,
        "matches": [],
        "current_step": 0,
        "conversation": "",
        "last_solution_text": "",
        "case_id": None,
        "awaiting_feedback": False,
        "case_created": False,
        "resolution_steps_count": 0,
        "prediction_shown": False,
        "prediction_accepted": False,
        "predicted_issues": [],
        "customer_phone": None,
        "rating_submitted": False,
        "selected_rating": None,
        "feedback_submitted": False
    })
    
    return (
        "",                 # name
        "",                 # phone
        "",                 # text
        None,               # audio ✔️ مسموح هنا فقط
        gr.update(visible=False, value=None),  # pred_radio
        "",                 # out_issue
        "",                 # out_crm
        "",                 # out_response
        gr.update(visible=False),
        gr.update(visible=False),
        ""                  # rating_value
    )

def full_reset_and_refresh(sub, status, year, month):
    """Reset session and refresh dashboard"""
    agent_outputs = reset_session()
    dash_outputs = dashboard_view(sub, status, year, month)
    return agent_outputs + dash_outputs

# ==========================================
# 🔐 ADMIN FUNCTIONS
# ==========================================
def enable_admin(pwd):
    """Enable admin mode"""
    if pwd == ADMIN_PASSWORD:
        return gr.update(visible=True), "✅ تم تفعيل وضع الأدمن"
    return gr.update(visible=False), "❌ كلمة المرور غير صحيحة"

def admin_retrain():
    """Admin retrain function"""
    global IS_TRAINING
    
    if IS_TRAINING:
        return "⏳ التدريب جاري بالفعل، انتظر..."
    
    IS_TRAINING = True
    try:
        result = retrain_behavioral_model()
        return f"✅ {result}"
    except Exception as e:
        return f"❌ خطأ: {e}"
    finally:
        IS_TRAINING = False

# ==========================================
# 🚀 UNIFIED GRADIO INTERFACE
# ==========================================
with gr.Blocks(theme=gr.themes.Base(), css=UNIFIED_CSS) as demo:
    
    # Tab 1: AI Service Agent
    with gr.Tab("📞 AI Service Agent"):
        gr.Markdown("""### 🤖 Smart Telecom Agent
<small style="color:#64748b;">
نظام ذكي لإدارة شكاوى عملاء الاتصالات وتحليلها تلقائيًا
</small>
""")
        
        with gr.Row():
            # =====================
            # Input Column
            # =====================
            with gr.Column(scale=1):
                name = gr.Textbox(label="👤 اسم العميل", placeholder="اكتب اسم العميل")
                phone = gr.Textbox(label="📱 رقم الهاتف", placeholder="01XXXXXXXXX")
                
                with gr.Row():
                    add_customer_btn = gr.Button("➕ إضافة / تأكيد بيانات العميل", variant="secondary", scale=1)
                
                text = gr.Textbox(label="📝 الشكوى", placeholder="اكتب المشكلة هنا أو استخدم الصوت", lines=3)
                audio = gr.Audio(type="filepath", label="🎙️ تسجيل صوتي (اختياري)", streaming=False)
                
                pred_radio = gr.Radio(
                    label="🔮 بناءً على سجلك السابق، اختر المشكلة المتوقعة:",
                    choices=[],
                    visible=False
                )
                
                btn = gr.Button("🔍 تحليل المشكلة", variant="primary")
                
                gr.Markdown("---")
                
                # Feedback Section
                with gr.Column(visible=False) as feedback_container:
                    gr.Markdown("### ❓ هل تم حل المشكلة؟")
                    with gr.Row():
                        solved_btn = gr.Button("✅ اتحلت", elem_classes="feedback-btn-solved")
                        not_solved_btn = gr.Button("❌ لم تتحل", elem_classes="feedback-btn-notsolved")
                
                # Rating Section
                with gr.Column(visible=False, elem_classes="rating-container") as rating_container:
                    gr.Markdown("### ⭐ تقييم الخدمة (اختياري)")
                    gr.Markdown("اختر عدد النجوم من 1 (سيء) إلى 5 (ممتاز)")
                    
                    with gr.Row(elem_classes="star-rating"):
                        star1 = gr.Button("1", elem_classes="star-btn")
                        star2 = gr.Button("2", elem_classes="star-btn")
                        star3 = gr.Button("3", elem_classes="star-btn")
                        star4 = gr.Button("4", elem_classes="star-btn")
                        star5 = gr.Button("5", elem_classes="star-btn")
                    
                    with gr.Row():
                        submit_rating_btn = gr.Button("📤 إرسال التقييم", variant="primary")
                        skip_rating_btn = gr.Button("⏭ تخطي التقييم", variant="secondary", elem_classes="no-rating-btn")
                    
                    rating_value = gr.Textbox(visible=False)
                
                end_btn = gr.Button("🔄 بدء حالة جديدة", elem_classes="stop-btn")
            
            # =====================
            # Output Column
            # =====================
            with gr.Column(scale=1):
                out_issue = gr.Textbox(
                    label="📌 المشكلة المسجلة",
                    value="",
                    interactive=False
                )
                out_crm = gr.Textbox(
                    label="👤 بيانات العميل (CRM)",
                    value="سيتم عرض بيانات العميل بعد إدخال رقم الهاتف",
                    interactive=False,
                    lines=5
                )
                out_response = gr.Textbox(
                    label="🧠 سجل الحلول / المحادثة",
                    value="👋 أدخل رقم الهاتف للبدء",
                    lines=14,
                    interactive=False
                )
        
        # =====================
        # EVENT HANDLERS
        # =====================
        
        # Confirm customer button
        add_customer_btn.click(
            confirm_customer,
            inputs=[name, phone],
            outputs=[pred_radio, out_response, feedback_container, out_crm]
        )
        
        
        
        # Main analysis button
        btn.click(
            pipeline,
            inputs=[name, phone, text, audio, pred_radio],
            outputs=[out_issue, out_crm, out_response, feedback_container, rating_container, rating_value]
        )
        
        # Feedback buttons
        solved_btn.click(
            handle_solved,
            outputs=[out_response, feedback_container, rating_container]
        )
        
        not_solved_btn.click(
            handle_not_solved,
            outputs=[out_response, feedback_container, rating_container]
        )
        
        # Star rating buttons
        star_buttons = [star1, star2, star3, star4, star5]
        for i, star_btn in enumerate(star_buttons, 1):
            star_btn.click(
                lambda x=i: str(x),
                outputs=[rating_value]
            )
        
        # Submit rating button
        submit_rating_btn.click(
            submit_rating,
            inputs=[rating_value],
            outputs=[out_response, rating_container]
        )
        
        # Skip rating button
        skip_rating_btn.click(
            skip_rating,
            outputs=[out_response, rating_container]
        )
    
    # -------------------------------------
    # TAB 2: ANALYTICS DASHBOARD
    # -------------------------------------
    with gr.Tab("📊 Analytics Dashboard"):
        # Header
        with gr.Row(elem_classes="header-box"):
            gr.HTML(f"""
                <div style="display: flex; align-items: center; gap: 20px;">
                    <img src="{LOGO_HTML_SRC}" style="width: 80px; height: 80px; border-radius: 15px; object-fit: cover;">
                    <div>
                        <h1 style="margin:0; color: #4ecca3; font-size: 28px;">VoiceCare AI</h1>
                        <p style="margin:0; color: #a0a0c0; font-size: 14px;">Live Analytics Dashboard • Telecom Solution</p>
                    </div>
                </div>
            """)
        
        # Filters
        with gr.Row():
            # Load data for filter options
            df_init = load_data()
            
            sub_opts = ["All"]
            if not df_init.empty and 'subscription_type' in df_init.columns:
                sub_opts += sorted(df_init["subscription_type"].dropna().unique().tolist())
            
            year_opts = ["All"]
            if not df_init.empty and 'created_at' in df_init.columns:
                years = df_init["created_at"].dt.year.dropna().unique()
                if len(years) > 0:
                    year_opts += sorted(years.astype(int).astype(str).tolist())
            
            sub_filter = gr.Dropdown(sub_opts, value="All", label="نوع الباقة")
            status_filter = gr.Dropdown(["All", "اتحلت", "قيد المعالجة", "مصعد للدعم"], value="All", label="حالة الحل")
            year_filter = gr.Dropdown(year_opts, value="All", label="السنة")
            month_filter = gr.Dropdown(
                ["All"] + [str(i) for i in range(1, 13)],
                value="All",
                label="الشهر"
            )
        
        # ======================================
        # 🔐 ADMIN PANEL (Hidden)
        # ======================================
        with gr.Accordion("🔐 Admin Panel", open=False):
            admin_pass = gr.Textbox(
                label="Admin Password",
                type="password",
                placeholder="Enter admin password"
            )
            admin_status = gr.Textbox(
                label="Status",
                interactive=False
            )
            retrain_btn = gr.Button(
                "🔁 إعادة تدريب النموذج",
                variant="stop",
                visible=False
            )
            admin_pass.submit(
                enable_admin,
                inputs=admin_pass,
                outputs=[retrain_btn, admin_status]
            )
            retrain_btn.click(
                admin_retrain,
                outputs=admin_status
            )
        
        # KPI Cards
        with gr.Row():
            total_v = gr.Number(label="إجمالي الحالات", elem_classes="kpi-card")
            rate_v = gr.Textbox(label="معدل الحل", elem_classes="kpi-card")
            rating_v = gr.Number(label="متوسط التقييم", elem_classes="kpi-card")
            time_v = gr.Textbox(label="متوسط وقت الحل", elem_classes="kpi-card")
            pred_shown_v = gr.Number(label="تم عرض التنبؤ", elem_classes="kpi-card")
            pred_rate_v = gr.Textbox(label="دقة التنبؤ", elem_classes="kpi-card")
        
        # Charts
        with gr.Row():
            chart1 = gr.Plot(label="الاتجاه الأسبوعي")
            chart2 = gr.Plot(label="توزيع الحالات")
            chart3 = gr.Plot(label="تحليل التقييمات")
        
        # Data Table
        with gr.Row():
            table_view = gr.Dataframe(
                interactive=False,
                label="سجل الحالات التفصيلي",
                wrap=True,
                datatype="str"
            )
        
        # Dashboard Logic Binding
        dash_outputs = [
            total_v, rate_v, rating_v, time_v,
            pred_shown_v, pred_rate_v,
            chart1, chart2, chart3, table_view
        ]
        dash_inputs = [sub_filter, status_filter, year_filter, month_filter]
        
        for comp in dash_inputs:
            comp.change(dashboard_view, dash_inputs, dash_outputs)
        
        # Reset button
        end_btn.click(
            full_reset_and_refresh,
            inputs=dash_inputs,
            outputs=[
                name, phone, text, audio, pred_radio,
                out_issue, out_crm, out_response,
                feedback_container, rating_container, rating_value
            ] + dash_outputs
        )
        
        # Initial load
        demo.load(
            fn=lambda: dashboard_view("All", "All", "All", "All"),
            inputs=[],
            outputs=dash_outputs
        )

# ==========================================
# 🟢 MAIN LAUNCH
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting VoiceCare AI Application...")
    print(f"📊 Dashboard available at: http://0.0.0.0:7860")
    demo.launch(
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
        share=True

    )