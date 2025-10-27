from ultralytics import YOLO
import cv2
import os

# Load pre-trained YOLOv8
model = YOLO('yolov8n.pt')  # Auto-downloads (~6MB)

# Input video
input_path = '/pscratch/sd/m/mpoona/nfl_project/train/58168_003392_Sideline.mp4'  # Update if different
output_dir = '/pscratch/sd/m/mpoona/nfl_project/output'
os.makedirs(output_dir, exist_ok=True)

# Run inference
results = model.predict(
    source=input_path,
    conf=0.25,  # Confidence threshold
    iou=0.45,   # NMS IoU threshold
    save=True,  # Save video with boxes
    save_txt=True,  # Save bounding box txt
    project=output_dir,
    name='baseline'
)

# Save sample frames for inspection
for r in results[:5]:
    img = r.plot()  # Draw boxes/labels
    cv2.imwrite(f'{output_dir}/baseline/frame_{r.frame_idx:04d}.jpg', img)
