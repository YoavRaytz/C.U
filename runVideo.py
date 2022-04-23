
import os
import time
from collections import OrderedDict
from glob import glob
from pprint import pprint

import cv2
# import IPython.display
import matplotlib.pyplot as plt
import numpy as np
# import skimage
import torch
from PIL import Image

import clip
from emotions_detection import FER_model
frame_rate = 10
prev = 0
d = torch.load('3_templates_3_labels_1.pt')

model = FER_model(d)
cap = cv2.VideoCapture(0)

writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

try:
    while True:
        time_elapsed = time.time() - prev

        _, img = cap.read()
        # if time_elapsed > 1./frame_rate:
        #     prev = time.time()
        out = model.infer(img)

        detections = [out['emotions'][j][0] for j in range(len(out['emotions']))]
        first, second, third = detections

        gender = out['gender']

        cv2.rectangle(img, (0,0), (120,200), (255,150,50), -1)

        cv2.putText(img, gender, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2 )

        cv2.putText(img, first, (0,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2 )
        cv2.putText(img, second, (0,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2 )
        cv2.putText(img, third, (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2 )

        
        cv2.imshow('img', img)
        writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:

    print(e)

print("free writer")
writer.release()
print("done free writer")
print("free cap")
cap.release()
print("done free cap")
cv2.destroyAllWindows()