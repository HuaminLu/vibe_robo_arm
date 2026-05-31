import torch
import cv2


class ObjectDetector:
    def __init__(self, model_name: str = 'yolov5s', device: str | None = None, conf_thres: float = 0.3):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.conf = conf_thres

    def detect(self, frame):
        results = self.model(frame)
        detections = []

        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[int(cls)]
                detections.append({
                    'label': label,
                    'confidence': float(conf),
                    'box': (x1, y1, x2, y2),
                })

        return detections

    def draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            confidence = det['confidence']
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
