import io
from numpy import save, source
import torch
from PIL import Image

# Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
# img = Image.open("zidane.jpg")  # PIL image direct open

# Read from bytes as we do in app
with open("1.jpg", "rb") as file:
    img_bytes = file.read()
# with open("zidane.jpg", "rb") as file:
#     img_bytes = file.read()
img = Image.open(io.BytesIO(img_bytes))

results = model(img)  # includes NMS
crops=results.crop(save=True)

print(results.pandas().xyxy[0])
