import numpy as np
import cv2
from inference import  get_model
from inference.core.interfaces.stream.sinks import render_boxes
import supervision as sv
import os

fps_monitor = sv.FPSMonitor()
fps_monitor.reset()
fps_hist = []

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.25)

model = get_model("yolov8n-640", api_key=os.environ["ROBOFLOW_API_KEY"])

cap = cv2.VideoCapture('/opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    predictions = model.infer(frame)
    detections = sv.Detections.from_inference(predictions[0])
    
    fps_monitor.tick()
    fps = round(fps_monitor.fps,2)
    fps_hist.append(fps)
            
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.draw_text(scene=annotated_image, text=f"THROUGHPUT: {fps}FPS",text_anchor=sv.Point(x=100, y=400), text_color=sv.Color.WHITE)

    cv2.imshow("output", annotated_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


avg_fps = np.mean(np.array(fps_hist))
print(f"MEAN FPS:{avg_fps}")
