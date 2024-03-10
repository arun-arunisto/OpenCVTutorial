import streamlit as st
import cv2

st.set_page_config(
    page_title="Workout file"
)

st.title("Multiple Cameras")

cam1, cam2 = st.columns((1, 1))
video_1 = cam1.empty()
video_2 = cam2.empty()

cap1 = cv2.VideoCapture("videos/cars.mp4")
cap2 = cv2.VideoCapture("videos/motorbikes-1.mp4")

while cap1.isOpened() or cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    video_1.image(frame1, channels="RGB")
    video_2.image(frame2, channels="RGB")


cap1.release()
cap2.release()
cv2.destroyAllWindows()
