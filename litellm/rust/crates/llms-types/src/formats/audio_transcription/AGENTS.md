# rules

- The transcription contract follows OpenAI's `audio/transcriptions` shape, which is LiteLLM's normalized input and output for every transcription provider
- Providers without that endpoint translate in `llms/src/<provider>/audio_transcription/` (Bedrock builds a Converse request)
- Parameter support differs per provider and is filtered in the provider config, not by trimming the type

# references

- https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions
