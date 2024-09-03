# importing libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def detect(image):

    faces = face_cascade.detectMultiScale(
        image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)

    # Returning the image with bounding boxes drawn on it 
    return image, faces


# ui funct
def main():
    st.title("Face Detection App")

    st.header("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        # Reading image from video stream
        _, img = camera.read()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, a = detect(img)
       
        FRAME_WINDOW.image(img)
    else:
        camera.release()
        cv2.destroyAllWindows()
        st.write('Stopped')


if __name__ == "__main__":
    main()