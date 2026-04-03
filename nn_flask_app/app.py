from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# ======================
# Cấu hình thư mục upload
# ======================
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 32 → 16
        x = self.pool(torch.relu(self.conv2(x)))  # 16 → 8
        x = self.pool(torch.relu(self.conv3(x)))  # 8 → 4
        x = x.view(x.size(0), -1)                 # 2048
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ======================
# Load model
# ======================
device = torch.device("cpu")

model = CNN(num_classes=2)
model.load_state_dict(
    torch.load("model/best_catdog_cnn.pth", map_location=device)
)
model.eval()

class_names = ['Con mèo', 'Con chó']

# ======================
# ======================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# ======================
# Router chính
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                _, pred = torch.max(output, 1)
                result = class_names[pred.item()]

    return render_template("index.html",
                           result=result,
                           image=image_path)

# ======================
# Run app
# ======================
if __name__ == "__main__":
    app.run(debug=True)