import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import math
import cvzone
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import torch

# Optimized tracker class
class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = 'deep_sort/networks/mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return self.tracks

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()
        return self.tracks

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            tracks.append(Track(id, bbox))
        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox

torch.cuda.set_device(0)
model = YOLO("best.engine", task="detect")  # TensorRT optimized model

cap = cv2.VideoCapture('id4.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file 'id4.mp4'.")
    exit()

class_list = []
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

cy1 = 427
offset = 6
count = 0
tracker = Tracker()
tracker1 = Tracker()
tracker2 = Tracker()
tracker3 = Tracker()
bus = []
car = []
auto_rikshaw = []
motorcycle = []

# CUDA-optimized video frame reading and processing
gpu_frame = cv2.cuda_GpuMat()  # Create a GpuMat object for CUDA processing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Upload the frame to the GPU
    gpu_frame.upload(frame)

    # Resize the frame using CUDA (GPU)
    gpu_frame_resized = cv2.cuda.resize(gpu_frame, (1020, 500))
    frame = gpu_frame_resized.download()  # Download back to CPU if needed

    # YOLO inference
    results = model(frame, imgsz=800)
    a = results[0].boxes.data
    a_cpu = a.cpu().numpy()  # Convert the results to CPU
    px = pd.DataFrame(a_cpu).astype("float")

    list, list1, list2, list3 = [], [], [], []

    for index, row in px.iterrows():
        x1, y1, x2, y2, score = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4]
        d = int(row[5])
        c = class_list[d]

        if 'bus' in c:
            list.append([x1, y1, x2, y2, score])
        elif 'car' in c:
            list1.append([x1, y1, x2, y2, score])
        elif 'auto-rikshaw' in c:
            list2.append([x1, y1, x2, y2, score])
        elif 'motor-cycle' in c:
            list3.append([x1, y1, x2, y2, score])

    # Tracker updates
    bbox_idx = tracker.update(frame, list)
    bbox1_idx = tracker1.update(frame, list1)
    bbox2_idx = tracker2.update(frame, list2)
    bbox3_idx = tracker3.update(frame, list3)

    # Drawing bounding boxes and labels
    for track in bbox_idx:
        x3, y3, x4, y4 = track.bbox
        id = track.track_id
        if cy1 < (y3 + y4) // 2 + offset and cy1 > (y3 + y4) // 2 - offset:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            if bus.count(id) == 0:
                bus.append(id)

    # Similarly for cars, auto-rikshaws, and motorcycles...

    # Display counts and lines
    cvzone.putTextRect(frame, f'Bus: {len(bus)}', (50, 120), scale=2, thickness=2, colorR=(0, 0, 255))
    cv2.line(frame, (405, 427), (580, 427), (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
