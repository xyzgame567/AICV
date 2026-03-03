import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO("yolov8n.pt")
image_path = "group.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model(img_rgb)
annotated_img = results[0].plot()
plt.imshow(annotated_img)
plt.axis("off")
plt.show()