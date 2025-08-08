import cv2
import streamlit as st
import mediapipe as mp

# Set the page configuration
st.set_page_config(page_title="Phone AI Detection Test")
st.title("Live Player Detection")
st.write("Press START to use your phone's camera.")

# A container to display the detected position
position_placeholder = st.empty()

# The main class that handles the AI processing for each frame
class PoseDetector:
    def __init__(self):
        # Use the default configuration
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.detected_position = "Waiting..."

    def process_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img)
        h, w, _ = img.shape
        center_line = w // 2

        if results.pose_landmarks:
            nose_x = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].x * w
            if nose_x < center_line:
                self.detected_position = "Left"
            else:
                self.detected_position = "Right"
        else:
            self.detected_position = "No Player Detected"
        
        return img


# Use Streamlit's built-in camera input widget
image = st.camera_input("Take a picture")

if image:
    # Process the captured image with PoseDetector
    pose_detector = PoseDetector()
    img = pose_detector.process_frame(image)
    st.image(img, caption="Detected Player Position", channels="RGB")
    position_placeholder.markdown(
        f"## Position: **{pose_detector.detected_position}**"
    )
else:
    position_placeholder.markdown("## Waiting for camera input...")
