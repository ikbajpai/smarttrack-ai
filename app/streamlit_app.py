"""
streamlit_app.py
----------------
Placeholder for the SmartTrack AI Streamlit web application.

Responsibilities (Day 5):
- Sidebar controls: video source, confidence threshold, zone drawing
- Live video feed with annotated detections and zones
- Intrusion alert panel with real-time log table
- Session state management
"""

import streamlit as st


def main():
    st.set_page_config(
        page_title="SmartTrack AI",
        page_icon="🎯",
        layout="wide",
    )
    st.title("SmartTrack AI — Restricted Zone Intrusion Detection")
    st.info("Day 1: Project initialized. Detection pipeline coming on Day 2.")


if __name__ == "__main__":
    main()
