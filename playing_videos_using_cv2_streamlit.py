import streamlit as st
import cv2

st.set_page_config(
    page_title="Sample file"
)

st.title("Sample workout file")

cap = cv2.VideoCapture("videos/video.mp4")

video_frame = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    video_frame.image(frame, channels="BGR")

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()



