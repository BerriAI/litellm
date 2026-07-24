# OpenAI Text-to-Speech Guardrail Translation Handler

Handler for processing OpenAI's text-to-speech endpoint (`/v1/audio/speech`) with guardrails.

## Overview

This handler processes text-to-speech requests by:
1. Extracting the input text from the request
2. Applying guardrails to the input text
3. Updating the request with the guardrailed text
4. Returning the output unchanged (audio is binary, not text)

## Data Format

### Input Format

```json
{
  "model": "tts-1",
  "input": "The quick brown fox jumped over the lazy dog.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

### Output Format

The output is binary audio data (MP3, WAV, etc.), not text, so it cannot be guardrailed.

## Usage

The handler is automatically discovered and applied when guardrails are used with the text-to-speech endpoint.

### Example: Using Guardrails with Text-to-Speech

```bash
curl -X POST 'http://localhost:4000/v1/audio/speech' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "tts-1",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy",
    "guardrails": ["content_moderation"]
}' \
--output speech.mp3
```

The guardrail will be applied to the input text before the text-to-speech conversion.

### Example: PII Masking in TTS Input

```bash
curl -X POST 'http://localhost:4000/v1/audio/speech' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "tts-1",
    "input": "Please call John Doe at john@example.com",
    "voice": "nova",
    "guardrails": ["mask_pii"]
}' \
--output speech.mp3
```

The audio will say: "Please call [NAME_REDACTED] at [EMAIL_REDACTED]"

### Example: Content Filtering Before TTS

```bash
curl -X POST 'http://localhost:4000/v1/audio/speech' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "tts-1-hd",
    "input": "This is the text that will be spoken",
    "voice": "shimmer",
    "guardrails": ["content_filter"]
}' \
--output speech.mp3
```

## Implementation Details

### Input Processing

- **Field**: `input` (string)
- **Processing**: Applies guardrail to input text
- **Result**: Updated input text in request

### Output Processing

- **Processing**: Not applicable (audio is binary data)
- **Result**: Response returned unchanged

## Use Cases

1. **PII Protection**: Remove personally identifiable information before converting to speech
2. **Content Filtering**: Remove inappropriate content before TTS conversion
3. **Compliance**: Ensure text meets requirements before voice synthesis
4. **Text Sanitization**: Clean up text before audio generation

## Extension

Override these methods to customize behavior:

- `process_input_messages()`: Customize how input text is processed
- `process_output_response()`: Currently a no-op, but can be overridden if needed

## Supported Call Types

- `CallTypes.speech` - Synchronous text-to-speech
- `CallTypes.aspeech` - Asynchronous text-to-speech

## Notes

- Only the input text is processed by guardrails
- Output processing is a no-op since audio cannot be text-guardrailed
- Both sync and async call types use the same handler
- Works with all TTS models (tts-1, tts-1-hd, etc.)
- Works with all voice options

## Common Patterns

### Remove PII Before TTS

```python
import litellm
from pathlib import Path

speech_file_path = Path(__file__).parent / "speech.mp3"
response = litellm.speech(
    model="tts-1",
    voice="alloy",
    input="Hi, this is John Doe calling from john@company.com",
    guardrails=["mask_pii"],
)
response.stream_to_file(speech_file_path)
# Audio will have PII masked
```

### Content Moderation Before TTS

```python
import litellm
from pathlib import Path

speech_file_path = Path(__file__).parent / "speech.mp3"
response = litellm.speech(
    model="tts-1-hd",
    voice="nova",
    input="Your text here",
    guardrails=["content_moderation"],
)
response.stream_to_file(speech_file_path)
```

### Async TTS with Guardrails

```python
import litellm
import asyncio
from pathlib import Path

async def generate_speech():
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = await litellm.aspeech(
        model="tts-1",
        voice="echo",
        input="Text to convert to speech",
        guardrails=["pii_mask"],
    )
    response.stream_to_file(speech_file_path)

asyncio.run(generate_speech())
```

