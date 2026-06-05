Abstraction / Routing logic for OpenAI's `/v1/realtime` endpoints.

## Realtime transcription (`gpt-realtime-whisper`)

`gpt-realtime-whisper` is the low-latency streaming speech-to-text model. It is a
Realtime transcription session, not the file-based `/audio/transcriptions` path. Use
the standard `gpt-4o-transcribe` / `whisper-1` models for request/response or file
transcription; use `gpt-realtime-whisper` for live streaming transcript deltas.

Both OpenAI and Azure OpenAI (Microsoft Foundry) are supported. Cost is tracked by input
audio duration (OpenAI: $0.017/minute), derived from the
`conversation.item.input_audio_transcription.completed` usage events.

### WebSocket

Connect to the proxy realtime WebSocket with `intent=transcription`, then send a
`session.update` configuring a transcription session:

```
wss://<proxy>/v1/realtime?model=gpt-realtime-whisper&intent=transcription
```

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": { "type": "audio/pcm", "rate": 24000 },
        "transcription": { "model": "gpt-realtime-whisper", "language": "en" }
      }
    }
  }
}
```

Append audio with `input_audio_buffer.append`, then `input_audio_buffer.commit` (when not
using server VAD). Listen for `conversation.item.input_audio_transcription.delta` and
`.completed` events. The proxy does not auto-trigger `response.create` for transcription
sessions.

### Ephemeral transcription session (WebRTC)

`POST /v1/realtime/transcription_sessions` mints an ephemeral session for browser/WebRTC
clients. The returned `client_secret.value` is encrypted by the proxy and exchanged via
`POST /v1/realtime/calls`.

```bash
curl https://<proxy>/v1/realtime/transcription_sessions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "input_audio_format": "pcm16",
        "input_audio_transcription": { "model": "gpt-realtime-whisper", "language": "en" }
      }'
```

For Azure, route to an `azure/gpt-realtime-whisper` deployment; the proxy targets
`/openai/realtime/transcription_sessions?api-version=...` and forwards
`intent=transcription` on the WebSocket.