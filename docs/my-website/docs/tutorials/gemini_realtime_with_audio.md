# Call Gemini Realtime API with Audio Input/Output 

:::info
Requires LiteLLM Proxy v1.70.1+
:::

1. Setup config.yaml for LiteLLM Proxy 

```yaml
model_list:
  - model_name: "gemini-2.0-flash"
    litellm_params:
      model: gemini/gemini-2.0-flash-live-001
    model_info:
      mode: realtime
```

2. Start LiteLLM Proxy 

```bash
litellm-proxy start
```

3. Run test script 

```python
import asyncio
import websockets
import json
import base64
from dotenv import load_dotenv
import wave
import base64
import soundfile as sf
import sounddevice as sd
import io
import numpy as np

# Load environment variables

OPENAI_API_KEY = "sk-1234"  # Replace with your LiteLLM API key
OPENAI_API_URL = 'ws://{PROXY_URL}/v1/realtime?model=gemini-2.0-flash' # REPLACE WITH `wss://{PROXY_URL}/v1/realtime?model=gemini-2.0-flash` for secure connection
WAV_FILE_PATH = "/path/to/audio.wav"  # Replace with your .wav file path

async def send_session_update(ws):
    session_update = {
        "type": "session.update",
        "session": {
            "conversation_id": "123456",
            "language": "en-US",
            "transcription_mode": "fast",
            "modalities": ["text"]
        }
    }
    await ws.send(json.dumps(session_update))

async def send_audio_file(ws, file_path):
    with wave.open(file_path, 'rb') as wav_file:
        chunk_size = 1024  # Adjust as needed
        while True:
            chunk = wav_file.readframes(chunk_size)
            if not chunk:
                break
            base64_audio = base64.b64encode(chunk).decode('utf-8')
            audio_message = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            await ws.send(json.dumps(audio_message))
            await asyncio.sleep(0.1)  # Add a small delay to simulate real-time streaming

    # Send end of audio stream message
    await ws.send(json.dumps({"type": "input_audio_buffer.end"}))

def play_base64_audio(base64_string, sample_rate=24000, channels=1):
    # Decode the base64 string
    audio_data = base64.b64decode(base64_string)

    # Convert to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Reshape if stereo
    if channels == 2:
        audio_np = audio_np.reshape(-1, 2)

    # Normalize
    audio_float = audio_np.astype(np.float32) / 32768.0

    # Play the audio
    sd.play(audio_float, sample_rate)
    sd.wait()


def combine_base64_audio(base64_strings):
    # Step 1: Decode base64 strings to binary
    binary_data = [base64.b64decode(s) for s in base64_strings]
    
    # Step 2: Concatenate binary data
    combined_binary = b''.join(binary_data)
    
    # Step 3: Encode combined binary back to base64
    combined_base64 = base64.b64encode(combined_binary).decode('utf-8')
    
    return combined_base64

async def listen_in_background(ws):
    combined_b64_audio_str = []
    try: 
        while True:
            response = await ws.recv()
            message_json = json.loads(response)
            print(f"message_json: {message_json}")

            if message_json['type'] == 'response.audio.delta' and message_json.get('delta'):
                play_base64_audio(message_json["delta"])
    except Exception: 
        print("END OF STREAM")

async def main():
    async with websockets.connect(
        OPENAI_API_URL,
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as ws:
        asyncio.create_task(listen_in_background(ws=ws))
        await send_session_update(ws)
        await send_audio_file(ws, WAV_FILE_PATH)

        

if __name__ == "__main__":
    asyncio.run(main())
```

