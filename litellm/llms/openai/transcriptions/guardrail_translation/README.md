# OpenAI Audio Transcription Guardrail Translation Handler

Handler for processing OpenAI's audio transcription endpoint (`/v1/audio/transcriptions`) with guardrails.

## Overview

This handler processes audio transcription responses by:
1. Applying guardrails to the transcribed text output
2. Returning the input unchanged (since input is an audio file, not text)

## Data Format

### Input Format

The input is an audio file, which cannot be guardrailed (it's binary data, not text).

```json
{
  "model": "whisper-1",
  "file": "<audio file>",
  "response_format": "json",
  "language": "en"
}
```

### Output Format

```json
{
  "text": "This is the transcribed text from the audio file."
}
```

Or with additional metadata:

```json
{
  "text": "This is the transcribed text from the audio file.",
  "duration": 3.5,
  "language": "en"
}
```

## Usage

The handler is automatically discovered and applied when guardrails are used with the audio transcription endpoint.

### Example: Using Guardrails with Audio Transcription

```bash
curl -X POST 'http://localhost:4000/v1/audio/transcriptions' \
-H 'Authorization: Bearer your-api-key' \
-F 'file=@audio.mp3' \
-F 'model=whisper-1' \
-F 'guardrails=["pii_mask"]'
```

The guardrail will be applied to the **output** transcribed text only.

### Example: PII Masking in Transcribed Text

```bash
curl -X POST 'http://localhost:4000/v1/audio/transcriptions' \
-H 'Authorization: Bearer your-api-key' \
-F 'file=@meeting_recording.mp3' \
-F 'model=whisper-1' \
-F 'guardrails=["mask_pii"]' \
-F 'response_format=json'
```

If the audio contains: "My name is John Doe and my email is john@example.com"

The transcription output will be: "My name is [NAME_REDACTED] and my email is [EMAIL_REDACTED]"

### Example: Content Moderation on Transcriptions

```bash
curl -X POST 'http://localhost:4000/v1/audio/transcriptions' \
-H 'Authorization: Bearer your-api-key' \
-F 'file=@audio.wav' \
-F 'model=whisper-1' \
-F 'guardrails=["content_moderation"]'
```

## Implementation Details

### Input Processing

- **Status**: Not applicable
- **Reason**: Input is an audio file (binary data), not text
- **Result**: Request data returned unchanged

### Output Processing

- **Field**: `text` (string)
- **Processing**: Applies guardrail to the transcribed text
- **Result**: Updated text in response

## Use Cases

1. **PII Protection**: Automatically redact personally identifiable information from transcriptions
2. **Content Filtering**: Remove or flag inappropriate content in transcribed audio
3. **Compliance**: Ensure transcriptions meet regulatory requirements
4. **Data Sanitization**: Clean up transcriptions before storage or further processing

## Extension

Override these methods to customize behavior:

- `process_output_response()`: Customize how transcribed text is processed
- `process_input_messages()`: Currently a no-op, but can be overridden if needed

## Supported Call Types

- `CallTypes.transcription` - Synchronous audio transcription
- `CallTypes.atranscription` - Asynchronous audio transcription

## Notes

- Input processing is a no-op since audio files cannot be text-guardrailed
- Only the transcribed text output is processed
- Guardrails apply after transcription is complete
- Both sync and async call types use the same handler
- Works with all Whisper models and response formats

## Common Patterns

### Transcribe and Redact PII

```python
import litellm

response = litellm.transcription(
    model="whisper-1",
    file=open("interview.mp3", "rb"),
    guardrails=["mask_pii"],
)

# response.text will have PII redacted
print(response.text)
```

### Async Transcription with Guardrails

```python
import litellm
import asyncio

async def transcribe_with_guardrails():
    response = await litellm.atranscription(
        model="whisper-1",
        file=open("audio.mp3", "rb"),
        guardrails=["content_filter"],
    )
    return response.text

text = asyncio.run(transcribe_with_guardrails())
```

