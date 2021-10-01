# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Object Detection - YOLOv3

# %%
import numpy as np
import cv2 as cv
from motrackers.detectors import YOLOv3
import pandas
import pandas as pd
import csv
COLS = ['id','bounding_box']
df = pd.DataFrame(columns=COLS)


# %%
VIDEO_FILE = "/Users/timseifert/multi-object-tracker/examples/video_data/people.mp4"
WEIGHTS_PATH = '/Users/timseifert/multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.weights'
CONFIG_FILE_PATH = '/Users/timseifert/multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.cfg'
LABELS_PATH = "/Users/timseifert/multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json"

USE_GPU = False
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2
DRAW_BOUNDING_BOXES = True


# %%
model = YOLOv3(
    weights_path=WEIGHTS_PATH,
    configfile_path=CONFIG_FILE_PATH,
    labels_path=LABELS_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    draw_bboxes=DRAW_BOUNDING_BOXES,
    use_gpu=USE_GPU
)


# %%
cap = cv.VideoCapture(VIDEO_FILE)


# %%
while True:
    ok, image = cap.read()
    
    if not ok:
        print("Cannot read the video feed.")
        break
    
    bboxes, confidences, class_ids = model.detect(image)
    updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
    iter = 0
    for cid in class_ids:
        try:
            label = "{}:{:.4f}".format(model.object_names[cid], confidences[iter])
            new_entry = []
            new_entry.append(label)
            new_entry.append(bboxes[iter])
            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)  
            df.to_csv('bounding_box_collection_first_ten.csv', columns=COLS,index=False) 
            iter+=1
        except:
             continue
    cv.imshow("image", updated_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyWindow("image")


# %%



