import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template_string

# ------------------ INIT ------------------
app = Flask(__name__)

model = models.resnet18(pretrained=True)
model.eval()

features = []

def hook_fn(module, input, output):
    features.append(output)

model.layer4.register_forward_hook(hook_fn)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ------------------ HEALTH SCORE ------------------
def compute_health_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:,:,1])
    brightness = np.mean(hsv[:,:,2])
    score = int((saturation + brightness) / 2)
    return max(0, min(100, score))

# ------------------ STAGE CLASSIFIER ------------------
def classify_stage(img):
    h, w, _ = img.shape

    # Focus on center crop
    crop = img[h//4:3*h//4, w//4:3*w//4]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Green
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    # White
    white_mask = cv2.inRange(hsv, (0, 0, 185), (180, 50, 255))
    white_ratio = np.sum(white_mask > 0) / white_mask.size

    # Pink (flower)
    pink_mask1 = cv2.inRange(hsv, (140, 40, 40), (180, 255, 255))
    pink_mask2 = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
    pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)
    pink_ratio = np.sum(pink_mask > 0) / pink_mask.size

    print("Pink:", round(pink_ratio,4),
          "White:", round(white_ratio,4),
          "Green:", round(green_ratio,4))

    if white_ratio > 0.10:
        return "Harvest Ready"
    elif white_ratio > 0.04:
        return "Bursting"
    elif pink_ratio > 0.003:
        return "Flowering"
    else:
        return "Vegetative"

# ------------------ GRAD CAM ------------------
def generate_gradcam(output, class_idx):
    model.zero_grad()
    score = output[0, class_idx]
    score.backward()

    grads = model.layer4[1].conv2.weight.grad
    fmap = features[-1].detach()[0]

    weights = torch.mean(grads, dim=(2,3))[0]
    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i,:,:]

    cam = np.maximum(cam.numpy(), 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()
    return cam

# ------------------ UI PAGE ------------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cotton Crop Analyzer</title>
        <style>
            body {
                font-family: Arial;
                background: #f2f4f7;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 12px;
                width: 480px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            }
            input { margin: 10px; }
            button {
                background: #2d7ff9;
                color: white;
                border: none;
                padding: 10px 25px;
                border-radius: 6px;
                cursor: pointer;
            }
            img {
                width: 100%;
                margin-top: 15px;
                border-radius: 8px;
            }
            .result {
                margin-top: 15px;
                background: #f0f0f0;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
            }
        </style>
    </head>

    <body>
        <div class="card">
            <h2>🌱 Cotton Crop Analyzer</h2>

            <form action="/analyze" method="post" enctype="multipart/form-data">
                <input type="file" name="image" required onchange="preview(event)">
                <br>
                <button type="submit">Analyze</button>
            </form>

            <img id="preview"/>

        </div>

        <script>
            function preview(event){
                document.getElementById("preview").src =
                    URL.createObjectURL(event.target.files[0]);
            }
        </script>
    </body>
    </html>
    """)

# ------------------ ANALYZE API ------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    img_pil = Image.open(file).convert("RGB")
    img_np = np.array(img_pil)

    img_tensor = transform(img_pil).unsqueeze(0)
    output = model(img_tensor)

    stage = classify_stage(img_np)
    is_ripped = True if stage in ["Bursting", "Harvest Ready"] else False
    health_score = compute_health_score(img_np)

    cam = generate_gradcam(output, torch.argmax(output).item())
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Encode GradCAM image for browser display
    _, buffer = cv2.imencode(".png", overlay)
    gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

    result = {
        "stage": stage,
        "is_ripped": is_ripped,
        "health_score": health_score
    }

    return render_template_string("""
        <div style="font-family:Arial; text-align:center; padding:30px">
            <h2>✅ Analysis Result</h2>

            <div class="result">
                <pre>{{ result }}</pre>
            </div>

            <h3>Grad-CAM Visualization</h3>
            <img src="data:image/png;base64,{{ gradcam }}" style="width:60%; border-radius:10px"/>

            <br><br>
            <a href="/">⬅ Upload Another Image</a>
        </div>
    """, result=result, gradcam=gradcam_b64)

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
