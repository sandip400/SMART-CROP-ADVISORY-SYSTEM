import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# Create model for 4 classes
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, 4)

# Load state dict but ignore mismatched layers
state_dict = torch.load("soil_model.pth", map_location=torch.device('cpu'))
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if "classifier.1" not in k:   # skip final layer
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)  # allow skipping
model.eval()
import torchvision.transforms as transforms
from PIL import Image

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
img_path = "taste2.jpg"   # change to your soil image
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

# Map index -> class name
class_names = ["Alluvial","Black","Red","Laterite"]  # keep same order as dataset
predicted_class = class_names[predicted.item()]

print("Predicted Soil Type:", predicted_class)

# Show image with label
plt.imshow(image)
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()
