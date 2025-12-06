import torchvision
from torchvision import transforms
from PIL import Image
import torch

model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

img = Image.open("gato.jpg")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    preds = model(x)

print(preds)