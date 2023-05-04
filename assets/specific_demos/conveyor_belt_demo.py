import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time ,sys
#from streamlit_embedcode import github_gist
import urllib.request
import urllib
#import moviepy.editor as moviepy
# from ..convenience import is_cv3


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = 'ObjectDetection/config_n_weights/yolov3.cfg'
weights_path = 'ObjectDetection/config_n_weights/yolov3.weights'
font_scale = 1
thickness = 1

url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
f = urllib.request.urlopen(url)
labels = [line.decode('utf-8').strip() for  line in f]
#f = open(r'C:\Users\Olazaah\Downloads\stream\labels\coconames.txt','r')
#lines = f.readlines()
#labels = [line.strip() for line in lines]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def cb_demo(si_stream, stop_stream, files):
    st.title('Video Based Object Detection')
    st.subheader('running detection on conveyor belt')
    st.write('---------------------------------------------------------')
    case_selector = st.radio('choose the use case',['Oranges','My own Video'])
    if case_selector == 'Oranges':
        orange_video = files[0]
        vid = orange_video
        st.video(orange_video)
    else:
        uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
        if uploaded_video != None:
            vid = uploaded_video.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_video.read()) # save video to disk

            st_video = open(vid,'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Uploaded Video")

    cap = cv2.VideoCapture(vid)
    _, image = cap.read()
    h, w = image.shape[:2]
    #out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc#(*'avc3'), fps, insize)
    frame_viewer = st.empty()
    orig, anot = st.columns(2)
    orig_cont = orig.empty()
    anot_cont = anot.empty()



    # fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    # out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
    count = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.text(f'# of frames in video {total}')
    # while True:
    for idx in range(total):
        _, image = cap.read()
        if _ != False:
            # st.text(count)
            orig_image = image
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.perf_counter()
            layer_outputs = net.forward(ln)
            time_took = time.perf_counter() - start
            count +=1
            # print(f"Time took: {count}", time_took)
            print(count)
            boxes, confidences, class_ids = [], [], []

            # loop over each of the layer outputs
            for output in layer_outputs:
                # loop over each of the object detections
                for detection in output:
                    # extract the class id (label) and confidence (as a probability) of
                    # the current object detection
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # discard weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > CONFIDENCE:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # perform the non maximum suppression given the scores defined before
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

            font_scale = 0.6
            thickness = 1

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in colors[class_ids[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                    text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                    # calculate text width & height to draw the transparent boxes as background of the text
                    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = x
                    text_offset_y = y - 5
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                    # add opacity (transparency to the box)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                    # now put the text (label: confidence %)
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

            # out.write(image)
            with frame_viewer.container():
                st.text(f'frame {idx} out of {total}')
            with orig_cont.container():
                st.image(orig_image)
            with anot_cont.container():
                st.image(image)
            # cv2.imshow("image", image)
            # st.image(image)
            
            if ord("q") == cv2.waitKey(1):
                break
        else:
            break


    #return "detected_video.mp4"
        
    cap.release()
    cv2.destroyAllWindows()
    # return "detected_video.mp4"


