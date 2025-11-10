import cv2
from ultralytics import YOLO
import time
import torch

# ========================= CONFIG =========================
MODEL_PATH = "best.pt"          # your trained weights
CONF_THRESH = 0.5               # confidence threshold
CLASSES = ["bendover", "jump", "lying", "run", "sit", "squat", "stand", "stretch", "walk"]

# Auto-select device
DEVICE = "0" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")

# ==========================================================
def main():
    print("🚀 Loading YOLOv11 model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return
    
    print("🎥 Live pose detection started — press 'q' to quit.\n")

    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)
        
        annotated = frame.copy()
        for r in results:
            boxes = r.boxes.xyxy
            cls_ids = r.boxes.cls
            confs = r.boxes.conf

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                cls_id = int(cls_ids[i])
                conf = float(confs[i])
                if cls_id < 0 or cls_id >= len(CLASSES):
                    continue

                label = f"{CLASSES[cls_id]} ({conf:.2f})"
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label at bottom of box
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_label = y2 + th + 6 if y2 + th + 6 < annotated.shape[0] else y2 - 10
                cv2.rectangle(annotated, (x1, y_label - th - 6), (x1 + tw + 6, y_label), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1 + 3, y_label - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS overlay
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("POLAR YOLOv11 - Live Pose Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stream ended.")


# ==========================================================
if __name__ == "__main__":
    main()
