import streamlit as st
import numpy as np
import sounddevice as sd
import asyncio
from websockets.sync.client import connect
import json
import queue
import threading

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Smaller chunks for lower latency
BACKEND_WS_URL = "ws://localhost:8001/ws/transcribe"

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.recording = False

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            callback=self.callback
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

def websocket_sender(audio_recorder, stop_event):
    with connect(BACKEND_WS_URL) as websocket:
        while not stop_event.is_set():
            try:
                # Get audio chunk from queue
                audio_chunk = audio_recorder.audio_queue.get(timeout=0.1)
                
                # Send to backend
                websocket.send(json.dumps({
                    "audio": audio_chunk.tobytes(),
                    "sample_rate": SAMPLE_RATE
                }))
                
                # Receive transcription
                transcription = websocket.recv()
                if transcription:
                    st.session_state.transcription += " " + transcription
                    
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"WebSocket error: {str(e)}")
                break

def main():
    st.title("Real-Time Whisper Transcription")
    st.write("Continuous microphone transcription using WebSockets")

    # Initialize session state
    if "transcription" not in st.session_state:
        st.session_state.transcription = ""
    if "recording" not in st.session_state:
        st.session_state.recording = False

    # Create audio recorder
    audio_recorder = AudioRecorder()
    stop_event = threading.Event()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording") and not st.session_state.recording:
            st.session_state.recording = True
            st.session_state.transcription = ""
            audio_recorder.start_recording()
            
            # Start WebSocket sender in a separate thread
            stop_event.clear()
            ws_thread = threading.Thread(
                target=websocket_sender,
                args=(audio_recorder, stop_event)
            )
            ws_thread.start()
            st.rerun()

    with col2:
        if st.button("Stop Recording") and st.session_state.recording:
            st.session_state.recording = False
            audio_recorder.stop_recording()
            stop_event.set()
            st.rerun()

    # Display status
    status = "🔴 Recording" if st.session_state.recording else "⏹️ Stopped"
    st.write(f"Status: {status}")

    # Display live transcription
    st.subheader("Live Transcription")
    transcription_placeholder = st.empty()
    
    # Update the transcription in real-time
    while st.session_state.recording:
        transcription_placeholder.text_area(
            "Transcription",
            value=st.session_state.transcription,
            height=300,
            key="transcription_area"
        )
        st.rerun()

    # Final transcription display
    if not st.session_state.recording and st.session_state.transcription:
        transcription_placeholder.text_area(
            "Transcription",
            value=st.session_state.transcription,
            height=300,
            key="final_transcription"
        )

if __name__ == "__main__":
    main()