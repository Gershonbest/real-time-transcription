from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import whisper
import numpy as np
import torch
import torchaudio
from io import BytesIO
import asyncio
import logging

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
model = whisper.load_model("base")

async def process_audio_chunk(audio_data: bytes, sample_rate: int):
    try:
        # Convert bytes to numpy array
        audio_buffer = BytesIO(audio_data)
        audio_array = np.frombuffer(audio_buffer.getvalue(), dtype=np.float32)
        
        # Resample if needed
        if sample_rate != 16000:
            audio_array = torchaudio.functional.resample(
                torch.from_numpy(audio_array),
                sample_rate,
                16000
            ).numpy()
        
        # Transcribe
        result = model.transcribe(audio_array, fp16=torch.cuda.is_available())
        return result["text"]
    
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return None

@app.websocket("/ws/transcribe")
async def websocket_transcription(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_json()
            audio_data = bytes(data["audio"])
            sample_rate = data["sample_rate"]
            
            # Process and transcribe
            transcription = await process_audio_chunk(audio_data, sample_rate)
            
            if transcription:
                await websocket.send_text(transcription)
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close(code=1011)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)