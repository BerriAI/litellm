"""
Simple xAI Voice Agent using LiveKit SDK with LiteLLM Gateway

This example shows how to use LiveKit's xAI realtime plugin through LiteLLM proxy.
LiteLLM acts as a unified interface, allowing you to switch between xAI, OpenAI, 
and Azure realtime APIs without changing your agent code.
"""
import asyncio
import json
import os
import websockets

# Configuration
PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
MODEL = os.getenv("LITELLM_MODEL", "grok-voice-agent")


async def run_voice_agent():
    """
    Simple voice agent that:
    1. Connects to xAI realtime API through LiteLLM proxy
    2. Sends a user message
    3. Streams back the response
    """
    
    url = f"ws://{PROXY_URL.replace('http://', '').replace('https://', '')}/v1/realtime?model={MODEL}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print(f"🎙️  Connecting to voice agent...")
    print(f"   Model: {MODEL}")
    print(f"   Proxy: {PROXY_URL}")
    print()
    
    async with websockets.connect(url, additional_headers=headers) as ws:
        # Receive initial connection event
        initial = json.loads(await ws.recv())
        print(f"✅ Connected! Event: {initial['type']}\n")
        
        # Get user input
        user_message = input("💬 Your message: ").strip()
        if not user_message:
            user_message = "Tell me a fun fact about AI!"
        
        print(f"\n🤖 Sending to {MODEL}...\n")
        
        # Send user message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}]
            }
        }))
        
        # Request response
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        }))
        
        # Stream response
        print("🎤 Response: ", end='', flush=True)
        transcript = []
        
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                event = json.loads(msg)
                
                # Capture transcript deltas
                if event['type'] == 'response.output_audio_transcript.delta':
                    delta = event.get('delta', '')
                    if delta:
                        print(delta, end='', flush=True)
                        transcript.append(delta)
                
                # Done when response completes
                elif event['type'] == 'response.done':
                    break
        
        except asyncio.TimeoutError:
            pass
        
        print("\n")
        
        if transcript:
            print(f"✅ Complete response: {''.join(transcript)}")
        
        await ws.close()


def main():
    """Run the voice agent"""
    print("=" * 70)
    print("LiveKit xAI Voice Agent via LiteLLM Proxy")
    print("=" * 70)
    print()
    
    try:
        asyncio.run(run_voice_agent())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure LiteLLM proxy is running:")
        print(f"  litellm --config config.yaml --port 4000")


if __name__ == "__main__":
    main()
