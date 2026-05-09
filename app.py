from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import img_to_array  # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.vgg16 import preprocess_input  # pyright: ignore[reportMissingImports]
import os
import threading
import atexit

app = Flask(__name__)

# ✅ Save uploads inside static folder so Flask can display them
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# Load trained model
# -------------------------------
try:
    model = load_model('model_building_defects_vgg16.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print("⚠️ Model not found or failed to load.")
    print(e)
    model = None

labels = ['Crack', 'Flake', 'Roof problem']

camera = cv2.VideoCapture(0)
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

@atexit.register
def cleanup():
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()

# -------------------------------
# Prediction function
# -------------------------------
CONFIDENCE_THRESHOLD = 0.4

def predict_defect(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        preds = model.predict(image, verbose=0)
        confidence = float(np.max(preds))
        label_index = int(np.argmax(preds))
        label = labels[label_index]

        if confidence < CONFIDENCE_THRESHOLD:
            text = "No defect found"
        else:
            text = f"Defect: {label} ({confidence*100:.2f}%)"

        return text, confidence
    except Exception as e:
        print("Prediction Error:", e)
        return "Prediction Error", 0.0

# -------------------------------
# Flask routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    result_text = ""
    uploaded_image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            result_text = "⚠️ No file part in the request."
        else:
            file = request.files['file']
            if file.filename == '':
                result_text = "⚠️ No file selected."
            else:
                # ✅ Save uploaded file in static/uploads
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # ✅ Return only relative path for HTML template
                uploaded_image_path = f"uploads/{file.filename}"

                image = cv2.imread(filepath)
                result_text, _ = predict_defect(image)

    return render_template('home.html', result=result_text, uploaded_image=uploaded_image_path)

@app.route('/intro')
def introduction():
    return render_template('intro.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page for defect detection (used by navbar link)."""
    result_text = ""
    uploaded_image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            result_text = "⚠️ No file part in the request."
        else:
            file = request.files['file']
            if file.filename == '':
                result_text = "⚠️ No file selected."
            else:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # ✅ Same fix for upload page
                uploaded_image_path = f"uploads/{file.filename}"

                image = cv2.imread(filepath)
                result_text, _ = predict_defect(image)

    return render_template('upload.html', result=result_text, uploaded_image=uploaded_image_path)

# -------------------------------
# Webcam & human detection
# -------------------------------
def process_frame(frame):
    if model is None:
        cv2.putText(frame, "Model not loaded", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Human detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    text, confidence = predict_defect(frame)
    color = (0, 255, 255) if confidence >= CONFIDENCE_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def show_opencv_window():
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = process_frame(frame)
        cv2.imshow("AI Defect & Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cleanup()

# -------------------------------
# Run Flask + OpenCV popup
# -------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=show_opencv_window)
    t.daemon = True
    t.start()
    app.run(debug=True, port=8000)