import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import json
import kagglehub
import os
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download dataset (downloads once, then uses cache)
path = kagglehub.dataset_download(
    "aryashah2k/mobile-captured-pharmaceutical-medication-packages"
)

dataset_root = os.path.join(
    path,
    "Mobile-Captured Pharmaceutical Medication Packages"
)

dataset = ImageFolder(dataset_root)

class_names = dataset.classes

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(torch.load("medicine_classifier.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image):

    image = image.convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    confidence, pred = torch.max(probs,1)

    return (
        class_names[pred.item()],
        confidence.item()*100
    )