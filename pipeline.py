from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv
import cv2
import numpy as np
import os

fps_monitor = sv.FPSMonitor()
fps_monitor.reset()
fps_hist = []

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.25)

def on_prediction(
    predictions: dict,
    video_frame: VideoFrame,
) -> None:
    
    detections = sv.Detections.from_inference(predictions)
    
    fps_monitor.tick()
    fps = round(fps_monitor.fps,2)
    fps_hist.append(fps)

    annotated_image = bounding_box_annotator.annotate(scene=video_frame.image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.draw_text(scene=annotated_image, text=f"THROUGHPUT: {fps}FPS",text_anchor=sv.Point(x=100, y=400), text_color=sv.Color.WHITE)

    cv2.imshow("output", annotated_image)
    cv2.waitKey(1)

# create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="yolov8n-640", # set the model id to a yolov8x model with in put size 1280
    video_reference="/opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.mp4", # set the video reference (source of video), it can be a link/path to a video file, an RTSP stream url, or an integer representing a device id (usually 0 for built in webcams)
    on_prediction=on_prediction, # tell the pipeline object what to do with each set of inference by passing a function
    api_key=os.environ["ROBOFLOW_API_KEY"], # provide your roboflow api key for loading models from the roboflow api
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()

avg_fps = np.mean(np.array(fps_hist))
print(f"MEAN FPS:{avg_fps}")
