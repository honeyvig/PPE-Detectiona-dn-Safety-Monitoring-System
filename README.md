# PPE-Detectiona-n-Safety-Monitoring-System
Enhance our Pelco CCTV System by developing an advanced AI algorithm to detect safety equipment such as helmets, vests, and harnesses attached to vests, as well as monitor specific safety conditions. This project aims to ensure compliance with safety protocols in hazardous environments and improve overall safety measures.

Project Scope:

The AI solution will focus on the following:

• PPE Detection: Detect and track individuals missing required PPE, including helmets, vests, and improperly attached harnesses.
• Safety Condition Monitoring: Trigger alerts if:
• The door is closed.
• The indicator light is red.
• Human motion is detected inside the area.

These conditions will trigger real-time alerts to authorized personnel to facilitate immediate action and ensure proper safety protocols are followed. Additionally, the system will send email notifications to stakeholders in case of any non-compliance or critical safety issues.

The project is expected to last for one month, during which the developer will be responsible for leading the AI algorithm’s development and its integration into the existing Pelco CCTV System.

• Expertise in computer vision and machine learning.
• Strong programming skills and knowledge of image processing.
• Experience in developing AI solutions for real-time detection and safety monitoring.
• Ability to anticipate challenges and propose innovative solutions.

This is an exciting opportunity to contribute to a high-impact safety system.
=====================
Python-based solution to enhance the Pelco CCTV system with AI capabilities for PPE detection and safety condition monitoring. This implementation uses YOLOv5 for object detection, OpenCV for video processing, and integrates conditions like detecting door status, indicator lights, and human motion. The solution also triggers email notifications for safety violations.
Prerequisites

    Install Required Libraries:

pip install torch torchvision opencv-python numpy smtplib

Download YOLOv5: Clone the YOLOv5 repository:

    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt

    Pre-trained Weights: Use a pre-trained YOLOv5 model or fine-tune it for specific PPE detection tasks (helmets, vests, harnesses, doors, indicator lights).

Python Script

import cv2
import torch
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace 'best.pt' with your trained model

# Safety conditions
DOOR_STATUS = "closed"  # Assume initial state is closed
INDICATOR_LIGHT = "red"  # Assume initial state is red
MOTION_DETECTED = False  # Initialize motion detection state

# Email configuration
EMAIL_SENDER = "your_email@example.com"
EMAIL_PASSWORD = "your_password"
EMAIL_RECEIVER = "stakeholder@example.com"

def send_email(subject, body):
    """
    Send an email notification for safety violations.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def process_frame(frame):
    """
    Process each frame for PPE detection and safety condition monitoring.
    """
    global DOOR_STATUS, INDICATOR_LIGHT, MOTION_DETECTED

    # Run inference
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detection results

    # Draw detections on the frame
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls, name = row
        if conf > 0.5:  # Confidence threshold
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Safety condition checks
            if name == "door" and DOOR_STATUS != "closed":
                send_email("Safety Alert: Door Opened", "The door is not closed.")
                DOOR_STATUS = "closed"
            if name == "light" and INDICATOR_LIGHT != "red":
                send_email("Safety Alert: Indicator Light Red", "The indicator light is red.")
                INDICATOR_LIGHT = "red"
            if name == "person":
                MOTION_DETECTED = True
                send_email("Safety Alert: Human Motion Detected", "Human motion detected in a restricted area.")

    return frame

def main():
    """
    Main function to process video feed.
    """
    video_source = "sample_video.mp4"  # Replace with your CCTV video source
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for safety monitoring
        processed_frame = process_frame(frame)

        # Display frame
        cv2.imshow("Safety Monitoring", processed_frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

Key Features:

    PPE Detection:
        Detects and tracks helmets, vests, and harnesses.
        Flags missing PPE.

    Safety Condition Monitoring:
        Monitors door status (closed).
        Tracks indicator light (red).
        Detects human motion in restricted areas.

    Real-Time Alerts:
        Sends email notifications for safety violations.

    Customizable:
        Replace best.pt with your fine-tuned YOLOv5 model for specific object detection.

    Scalable:
        Easily integrate with the Pelco CCTV system by replacing video_source with the CCTV stream URL.

Next Steps:

    Training Data: Fine-tune the YOLOv5 model with a dataset of PPE and safety conditions.
    Integration: Replace the sample video with a live CCTV feed.
    Enhancements:
        Add audio alerts for real-time violations.
        Store detection logs in a database for compliance tracking.
        Use additional tools like MQTT or WebSockets for real-time updates.

This script serves as the foundation for a robust AI-powered safety monitoring system tailored for hazardous environments.
