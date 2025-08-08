import cv2
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set the page configuration
st.set_page_config(page_title="Phone AI Detection Test")
st.title("Live Player Detection")
st.write("Press START to use your phone's camera.")

# A container to display the detected position
position_placeholder = st.empty()

# The main class that handles the AI processing for each frame
class PoseDetector(VideoTransformerBase):
    def __init__(self):
        # Remove output_tensor_range argument, and just use the default configuration
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.detected_position = "Waiting..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        h, w, _ = img.shape
        center_line = w // 2
        cv2.line(img, (center_line, 0), (center_line, h), (255, 0, 0), 2)
        
        if results.pose_landmarks:
            nose_x = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].x * w
            if nose_x < center_line:
                self.detected_position = "Left"
            else:
                self.detected_position = "Right"
        else:
            self.detected_position = "No Player Detected"
            
        cv2.putText(img, self.detected_position, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return img
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        # You can also add additional STUN servers if needed
    ]
}

try:
    ctx = webrtc_streamer(
        key="pose-detection",
        video_processor_factory=PoseDetector,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_configuration
    )
except Exception as e:
    st.error(f"Error during WebRTC setup: {e}")


# Correctly check for ctx.state
if ctx and ctx.video_transformer and hasattr(ctx.state, 'playing') and ctx.state.playing:
    detected_position = ctx.video_transformer.detected_position
    position_placeholder.markdown(
        f"## Position: **{detected_position}**"
    )
else:
    position_placeholder.markdown("## Waiting for camera input...")

