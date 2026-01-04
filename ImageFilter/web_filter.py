import streamlit as st
import cv2
import numpy as np
# from isort.io import Empty
import os
base_dir = os.path.dirname(__file__)
prototxt = os.path.join(base_dir, "module8_with_pradip.py")
from prototxt import buttons as filter
# button =
st.title("IMAGE FILTER")
image = st.file_uploader("Choose a file", type=["jpg", "png"])
col1, col2 = st.columns(2)
if 'count' not in st.session_state:
    st.session_state['count'] = 0
# count =0
if image is not None:
    image_read = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(image_read, 1)
    # flower = cv2.imread('./Applications/Flowers.jpg')
    image = cv2.resize(image, (1000, 900))
    col1.image(image, channels="BGR")
    press = col1.button("Click here to filter")
    if press:
        button ='N'
        st.session_state['count'] += 1
        # press = False
        print(st.session_state['count'])

        ##################################
        # if st.session_state['count'] ==1:

        if st.session_state['count'] ==1:
            button = 'c'
        if st.session_state['count'] ==2:
            button = 'g'
        if st.session_state['count'] ==3:
            button = 's'
        if st.session_state['count'] ==4:
            button = 'f'
        if st.session_state['count'] ==5:
            button = 'd'
        if st.session_state['count'] ==6:
            button = 'b'
        if st.session_state['count'] ==7:
            button = 'l'
        if st.session_state['count'] ==8:
            button = 'n'
        if st.session_state['count'] ==9:
            button = 't'
        if st.session_state['count'] >9:
            st.session_state['count']=0
            button = 'r'

        filter_canny = filter(button, image)
        # filtered_image = cv2.imshow('image filter', filter_canny)

        col2.image(filter_canny, channels="BGR")
        success, img_filter_canny = cv2.imencode(".png", filter_canny)
        col2.download_button("download",
                           data=img_filter_canny.tobytes(), file_name="filter_image.png")

    cv2.waitKey(0)


