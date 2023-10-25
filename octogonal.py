

from ultralytics import YOLO
from PIL import Image
import cv2
import supervision as sv
import numpy as np
import argparse


import numpy as np
ZONE_POLYGON = np.array([
    [500, 0],
    [353, 353],
    [0, 500],
    [-353, 353],
    [-500, 0],
    [-353, -353],
    [0, -500],
    [353, -353],
    [500, 0]
], dtype=np.int32)

ZONE_POLYGON += 400
ZONE_POLYGON = (ZONE_POLYGON * 0.6).astype(np.int32)

#video path
cap = cv2.VideoCapture("/Users/enesdemirpence/Downloads/Silencero gun test (online-video-cutter.com).mp4")

#model path
model = YOLO("/Users/enesdemirpence/Downloads/best (11).pt")

def main(cap,model):
    zone_polygon = ZONE_POLYGON

    # Genişlik ve yükseklik değerlerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    zone = sv.PolygonZone(polygon = zone_polygon, frame_resolution_wh = (width, height))
    zone_annotator = sv.PolygonZoneAnnotator(zone = zone,
                                             color = sv.Color.red(),
                                             thickness = 7,
                                             text_thickness=7,
                                             text_scale=2)
    
    model = model
    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
        )


    while True:
        ret,frame = cap.read()
        
        if not ret:
            break
        
        result = model(frame)[0]
        
        detections = sv.Detections.from_ultralytics(result)
        
        labels = [
            f"{model.names[0]} {class_id:0.2f}"
            for _, confidence, class_id, _, _,
            in detections
        ]
        
        zone.trigger(detections = detections)
        
        frame = zone_annotator.annotate(scene = frame)            
        cv2.imshow("yolov8",frame)
        
        if cv2.waitKey(33)==27:
            break
if __name__ == "__main__":
    main(cap,model)
