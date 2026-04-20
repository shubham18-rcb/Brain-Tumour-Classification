import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_file, url_for
from reportlab.pdfgen import canvas
from io import BytesIO

app = Flask(__name__)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'brain_tumor_best.h5')
# MobileNetV2 uses 160x160 as defined in your training script
IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- PRECAUTIONS DATABASE ---
PR = {
    "Glioma": [
        "Consult a Neuro-oncologist immediately.",
        "Monitor for increased intracranial pressure (headaches, vomiting).",
        "Avoid heavy lifting or strenuous physical activity.",
        "Seizure precaution: Ensure a safe environment."
    ],
    "Meningioma": [
        "Schedule a follow-up MRI to track growth rate.",
        "Consult with a neurosurgeon for surgical evaluation.",
        "Report any sudden vision changes or hearing loss.",
        "Maintain a low-sodium diet to manage possible edema."
    ],
    "Pituitary": [
        "Consult an Endocrinologist for hormone level testing.",
        "Visual field testing is highly recommended.",
        "Monitor for symptoms like extreme fatigue or thirst.",
        "Avoid high-stress environments to manage cortisol levels."
    ]
}


def validate_and_predict(img_path):
    img = cv2.imread(img_path)
    if img is None: return "Scanned image not detected", None

    # Verification Logic (Medical Scan Heuristic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 10 or np.mean(gray) > 180:
        return "input only brain scan images", None

    # Preprocessing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_prep = cv2.resize(img_rgb, IMG_SIZE) / 255.0
    img_input = np.expand_dims(img_prep, axis=0)

    preds = model.predict(img_input)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    label = LABELS[class_idx]

    if label == "No Tumor":
        return "No tumor detected", {"type": label, "conf": round(confidence, 2),
                                     "precautions": ["Continue regular health checkups.",
                                                     "Maintain a healthy lifestyle."]}

    return "Tumor Detected", {
        "type": label,
        "conf": round(confidence, 2),
        "precautions": PRECAUTIONS.get(label, [])
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            path = os.path.join(BASE_DIR, 'uploads', file.filename)
            if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
            file.save(path)
            status, data = validate_and_predict(path)
            return render_template('index.html', status=status, data=data)
    return render_template('index.html')


@app.route('/report/<t_type>/<t_conf>')
def print_report(t_type, t_conf):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, 800, "NEURAL DIAGNOSTIC REPORT")
    p.setFont("Helvetica", 12)
    p.drawString(100, 770, f"Detection: {t_type}")
    p.drawString(100, 750, f"Confidence: {t_conf}%")
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Report.pdf")


if __name__ == '__main__':
    app.run(debug=True)