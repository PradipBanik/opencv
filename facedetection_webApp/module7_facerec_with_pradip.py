import cv2
import streamlit as st
import numpy as np
import os

def detect(img_faces, con_input):
    base_dir = os.path.dirname(__file__)
    prototxt = os.path.join(base_dir, "deploy.prototxt")
    modell = os.path.join(base_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    model = cv2.dnn.readNetFromCaffe(prototxt, modell)


    blob = cv2.dnn.blobFromImage(img_faces, 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > con_input:
            x, y, w, h = detections[0, 0, i, 3:7]
            start = (int(x * img_faces.shape[1]), int(y * img_faces.shape[0]))
            end = (int(w * img_faces.shape[1]), int(h * img_faces.shape[0]))
            cv2.rectangle(img_faces, start, end, (0, 255, 0), 2)

    return img_faces

st.title("Face Recognition")
img_faces = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

# st.image(img_faces)
con_input = st.slider("Confidence", 0, 100)
con_input = con_input/100

col1, col2 = st.columns(2)
# col1.write("This is column 1")
# col2.write("This is column 2")

if img_faces is not None:
    # Convert uploaded file to bytes
    file_bytes = np.asarray(bytearray(img_faces.read()), dtype=np.uint8)

    # Decode image using OpenCV
    img_faces = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_faces = cv2.resize(img_faces, (300, 300))

    col1.image(img_faces, channels="BGR")
    # img_faces = cv2.imread(img_faces)
    img_faces = detect(img_faces, con_input)
    col2.image(img_faces, channels="BGR")
    # Encode image as PNG or JPG
    success, img_faces = cv2.imencode(".png", img_faces)
    col2.download_button("Download file", data=img_faces.tobytes(), file_name="faces.png", mime="image/png")

    # cv2.imshow("Face Detection", img_faces)
    # cv2.waitKey(0)
# detect(img_faces, con_input):

# st.image(img_faces)
