# app.py
import os
import sqlite3
import textwrap
import re
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from fpdf2 import FPDF
from skimage.segmentation import mark_boundaries
from lime import lime_image
import shap
import os
import pandas as pd
import cv2
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# ---------------------- CONFIG ----------------------
MODEL_PATH = "."  # your trained SavedModel directory
IMG_SIZE = (50, 50)
data_dir = "train"  # must match training
CLASS_NAMES = os.listdir(data_dir) if os.path.exists(data_dir) else ["class_1", "class_2"]


# ---------------------- DATABASE ----------------------
DB_FILE = "patients.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
c.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        image_path TEXT,
        prediction TEXT,
        advice TEXT
    )
""")
conn.commit()

# ---------------------- LOAD CNN ----------------------
@st.cache_resource(show_spinner=False)
def load_cnn():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_cnn()

# ---------------------- LOAD LLM ----------------------
@st.cache_resource(show_spinner=False)
def load_llm():
    # LLM support disabled - using rule-based advice instead
    return None

llm = load_llm()

# ---------------------- HELPERS ----------------------
def preprocess_pil_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def batch_predict_fn(images):
    imgs = []
    for img in images:
        arr = img.astype(np.float32)
        if arr.max() > 1.1:
            arr = arr / 255.0
        pil = Image.fromarray(arr.astype(np.uint8)).resize(IMG_SIZE)
        arr = np.array(pil).astype(np.float32) / 255.0
        imgs.append(arr)
    batch = np.stack(imgs, axis=0)
    return model.predict(batch)

def generate_medical_advice(pred_class):
    advice_map = {
        "CAD-Normal": "No coronary abnormality detected. Maintain a healthy lifestyle and regular checkups.",
        "CAD-Sick": "Signs of coronary artery disease detected. Consult a cardiologist for further evaluation.",
        "Covid": "COVID-19 indicators detected. Isolate, monitor oxygen levels, and consult a physician.",
        "glioma_tumor": "Glioma tumor indicators detected. Immediate consultation with a neurologist is advised.",
        "meningioma_tumor": "Meningioma tumor indicators detected. Further MRI and specialist consultation required.",
        "pituitary_tumor": "Pituitary tumor indicators detected. Endocrinology and neurology consultation advised.",
        "no_tumor": "No brain tumor detected. Continue routine health monitoring.",
        "Normal-Xray": "Normal chest X-ray detected. No abnormalities found.",
        "Pneumonia-MRI": "Pneumonia indicators detected. Medical treatment and monitoring required."
    }

    advice = advice_map.get(
        pred_class,
        "Consult a qualified medical professional for further diagnosis."
    )

    return (
        f"Prediction: {pred_class}\n\n"
        f"Advice: {advice}\n\n"
        "⚠️ Disclaimer: This system provides AI-assisted analysis only and is NOT a medical diagnosis."
    )


def clean_text_for_pdf(text):
    text = "".join(c if c.isprintable() else "?" for c in text)
    text = re.sub(r'(\S{50,})', lambda m: ' '.join([m.group(0)[i:i+50] for i in range(0, len(m.group(0)), 50)]), text)
    return text

def save_pdf_report(username, image_path, pred_label, advice_text):
    pdf = FPDF()
    pdf.add_page()
    font_path = "DejaVuSans-Bold.ttf"
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.cell(200, 10, txt="Medical Image Analysis Report", ln=True, align="C")

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {username}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Prediction Result: {pred_label}", ln=True, align="L")
    pdf.ln(10)
    try:
        pdf.image(image_path, x=60, w=90)
        pdf.ln(85)
    except Exception:
        pdf.cell(200, 10, txt="[Scan Image Missing]", ln=True, align="C")
        pdf.ln(10)

    pdf.set_font("DejaVu", size=11)
    pdf.cell(200, 10, txt="AI Advice:", ln=True, align="L")
    page_width = pdf.w - 2 * pdf.l_margin
    safe_text = advice_text.replace("\n", " \n ")
    wrapped_lines = textwrap.wrap(safe_text, width=95, break_long_words=True, break_on_hyphens=True)
    for wline in wrapped_lines:
        pdf.multi_cell(page_width, 6, txt=wline)
    return pdf.output(dest="S").encode("latin-1")


# ---------------------- SESSION ----------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

st.set_page_config(page_title="Medical Image Analysis", layout="wide")

# ---------------------- SIDEBAR ----------------------
menu = st.sidebar.radio("Navigation", ["Login", "Register", "Dashboard"])

# ---------------------- PAGES ----------------------
if menu == "Register":
    st.header("📝 Create Account")
    ru = st.text_input("Choose username")
    rp = st.text_input("Choose password", type="password")
    if st.button("Register"):
        if not ru or not rp:
            st.error("Username and password cannot be empty")
        else:
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (ru, rp))
                conn.commit()
                st.success("✅ Registered! Go to Login.")
            except Exception:
                st.error("User already exists.")

elif menu == "Login":
    st.header("🔑 Login")
    lu = st.text_input("Username")
    lp = st.text_input("Password", type="password")
    if st.button("Login"):
        c.execute("SELECT password FROM users WHERE username=?", (lu,))
        row = c.fetchone()
        if row and row[0] == lp:
            st.session_state.logged_in = True
            st.session_state.username = lu
            st.success(f"Welcome, {lu}!")
            st.rerun()
        else:
            st.error("Invalid username/password")

else:  # Dashboard
    if not st.session_state["logged_in"]:
        st.warning("Please login first.")
    else:
        username = st.session_state["username"]
        st.subheader(f"📊 Dashboard — Welcome {username}")
        
        uploaded = st.file_uploader("📤 Upload a medical image (jpg/png)", type=["png", "jpg", "jpeg"])


        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded Scan", width=300)

            # ----------------- Prediction -----------------
            x = preprocess_pil_image(pil_img)
            preds = model.predict(np.expand_dims(x, axis=0))
            pred_idx = int(np.argmax(preds[0]))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = preds[0][pred_idx]
            st.success(f"Prediction: **{pred_label}** (confidence {confidence:.2f})")

            # ----------------- LIME Explanation -----------------
            num_features = st.slider("LIME: Number of features", min_value=2, max_value=10, value=5)
            try:
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    np.array(pil_img),
                    batch_predict_fn,
                    top_labels=5,
                    hide_color=0,
                    num_samples=300,
                )
                # Pick label safely
                if pred_idx in explanation.local_exp:
                    label_to_use = pred_idx
                else:
                    label_to_use = list(explanation.local_exp.keys())[0]

                temp, mask = explanation.get_image_and_mask(
                    label=label_to_use,
                    positive_only=True,
                    num_features=num_features,
                    hide_rest=False
                )
                lime_vis = mark_boundaries(temp / 255.0, mask)
            except Exception as e:
                st.warning(f"LIME explanation failed: {e}")
                lime_vis = np.array(pil_img)

            # ----------------- SHAP Explanation -----------------
            # ----------------- SHAP Explanation -----------------
            shap_transparency = st.slider("SHAP: Overlay transparency", 0.0, 1.0, 0.4, 0.05)
            try:
                # collect only image files
                image_extensions = (".png", ".jpg", ".jpeg")
                background_paths = []
                for root, dirs, files in os.walk(data_dir):
                    for f in files:
                        if f.lower().endswith(image_extensions):
                            background_paths.append(os.path.join(root, f))
                background_paths = background_paths[:20]

                background = np.stack([preprocess_pil_image(Image.open(p)) for p in background_paths], axis=0)
                explainer_shap = shap.GradientExplainer(model, background)

                # Preprocess uploaded image
                x_input = np.expand_dims(preprocess_pil_image(pil_img), axis=0)
                shap_values = explainer_shap.shap_values(x_input)

                # Create SHAP overlay
                shap_map = np.sum(np.abs(shap_values[0][0]), axis=-1)  # shape: 50x50
                shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
                shap_map = np.stack([shap_map]*3, axis=-1)  # make 3 channels

                # Resize SHAP map to match uploaded image size
                shap_map_img = Image.fromarray((shap_map*255).astype(np.uint8)).resize(pil_img.size)
                shap_map_resized = np.array(shap_map_img).astype(np.float32)/255.0

                # Original image normalized
                pil_img_arr = np.array(pil_img).astype(np.float32)/255.0

                shap_image = (1-shap_transparency)*pil_img_arr + shap_transparency*shap_map_resized
                shap_image = np.clip(shap_image, 0, 1)

            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
                shap_image = np.array(pil_img)


            # ----------------- Display Explanations -----------------
            col1, col2 = st.columns(2)
            with col1:
                st.image(lime_vis, caption="🔥 LIME Explanation", width=300)
            with col2:
                st.image(shap_image, caption="🌡️ SHAP Explanation", width=300)

            # ----------------- Automated Advice -----------------
            advice_text = generate_medical_advice(pred_label)
            st.subheader("🧾 Automated Advice")
            st.info(advice_text)

            # ----------------- Save Report & PDF -----------------
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            save_path = os.path.join(reports_dir, f"{username}_{uploaded.name}")
            pil_img.save(save_path)
            c.execute(
                "INSERT INTO reports (username, image_path, prediction, advice) VALUES (?, ?, ?, ?)",
                (username, save_path, pred_label, advice_text),
            )
            conn.commit()

            pdf_bytes = save_pdf_report(username, save_path, pred_label, advice_text)
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"report_{username}.pdf",
                mime="application/pdf"
            )

        # ----------------- Previous Reports -----------------
        st.markdown("---")
        st.subheader("📂 Previous Reports")
        rows = c.execute(
            "SELECT id, image_path, prediction FROM reports WHERE username=? ORDER BY id DESC",
            (username,)
        ).fetchall()
        if not rows:
            st.info("No previous reports.")
        else:
            for rid, ipath, pred in rows:
                st.write(f"Report ID: {rid} — Prediction: {pred}")
                try:
                    st.image(ipath, width=150)
                except Exception:
                    st.write("(Image missing)")

        # ----------------- Logout -----------------
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.success("Logged out.")
            st.rerun()
