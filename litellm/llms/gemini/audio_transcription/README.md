# Gemini transcription language hints

Gemini transcription accepts multiple language hints through the provider-specific `language_codes` parameter. The existing OpenAI-compatible `language="en"` parameter still works for a single hint

```python
import litellm

with open("sample.wav", "rb") as audio:
    transcript = litellm.transcription(
        model="gemini/gemini-3.5-transcribe",
        file=audio,
        language_codes=["en-US", "es-ES"],
    )
```

Through the LiteLLM proxy, send repeated `language_codes[]` multipart fields or a JSON-encoded list in `language_codes`:

```bash
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -F 'model=gemini/gemini-3.5-transcribe' \
  -F 'file=@sample.wav' \
  -F 'language_codes=["en-US", "es-ES"]'
```

`language_codes` takes precedence over `language` when supplied. An empty list enables automatic language detection, even if `language` is also set. Omit both parameters to use automatic detection as before

Bare language codes use the same BCP-47 normalization as `language`; explicit script and region variants are preserved. Invalid lists return an error instead of silently discarding hints. Word timestamps and speaker diarization can be used with multiple hints

These hints guide recognition; they do not restrict the transcript to the listed languages. See the [Gemini transcription documentation](https://ai.google.dev/gemini-api/docs/transcribe#language-detection-and-hints) for provider behavior
