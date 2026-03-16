import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

# 1. Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 2. Dataset
train_data = datasets.ImageFolder("archive/train_data", transform=transform)
val_data = datasets.ImageFolder("archive/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 3. Model
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))

# 4. Training loop (simplified)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 10 == 0:  # print every 10 batches
            print(f"Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"✅ Epoch [{epoch+1}/10] completed | Avg Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "soil_model.pth")
print("✅ Model saved as soil_model.pth")
