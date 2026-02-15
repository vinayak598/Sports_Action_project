import streamlit as st
import cv2
import tempfile
import time

from analytics import process_frame, reset_match

st.set_page_config(layout="wide")

st.title("🏆 AI Decision Support System for Sports Officials")

sport=st.sidebar.selectbox("Select Sport",["Football","Kabaddi"])
mode=st.sidebar.radio("Input Source",["Upload Video","Live Camera"])

speed=st.sidebar.slider("Video Speed",1.0,4.0,2.0)

# ⭐ Faster Playback
frame_skip=int(10/speed)

if frame_skip<2:
    frame_skip=2

WIDTH,HEIGHT=560,320


if mode=="Upload Video":

    uploaded=st.file_uploader("Upload Match Video")

    if uploaded:

        reset_match()

        tfile=tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        cap=cv2.VideoCapture(tfile.name)
        stframe=st.empty()

        count=0
        prev=0

        while cap.isOpened():

            ret,frame=cap.read()
            if not ret:
                break

            count+=1

            if count%frame_skip!=0:
                continue

            frame=cv2.resize(frame,(WIDTH,HEIGHT))

            annotated=process_frame(frame,sport)

            now=time.time()
            fps=int(1/(now-prev)) if prev else 0
            prev=now

            cv2.putText(annotated,f"FPS:{fps}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

            stframe.image(annotated,channels="BGR")

        cap.release()


else:

    if st.checkbox("Start Camera"):

        reset_match()

        cap=cv2.VideoCapture(0)
        stframe=st.empty()

        prev=0

        while True:

            ret,frame=cap.read()
            if not ret:
                break

            frame=cv2.resize(frame,(WIDTH,HEIGHT))

            annotated=process_frame(frame,sport,live=True)

            now=time.time()
            fps=int(1/(now-prev)) if prev else 0
            prev=now

            cv2.putText(annotated,f"FPS:{fps}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

            stframe.image(annotated,channels="BGR")

        cap.release()
