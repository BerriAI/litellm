# Self-host FunASR audio transcriptions behind LiteLLM

LiteLLM already routes OpenAI-compatible `/audio/transcriptions` providers through
`custom_openai`. Use this when you want the LiteLLM proxy to expose a local
FunASR or SenseVoice speech-to-text server without adding a new LiteLLM provider.

## Start a FunASR-compatible transcription server

Install FunASR and the web server dependencies on the machine that will run ASR:

```bash
python -m pip install -U "funasr>=1.3.20" vllm fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda --port 8000
```

The server exposes an OpenAI-compatible endpoint at:

```text
http://localhost:8000/v1/audio/transcriptions
```

Use `--device cpu` for a CPU-only smoke test, or put the FunASR server on a GPU
host and point LiteLLM at that host.

## Configure LiteLLM

Add a `custom_openai` model to the proxy config. Set `api_base` to the OpenAI
base URL, not the full transcription path.

```yaml
model_list:
  - model_name: funasr-sensevoice
    litellm_params:
      model: custom_openai/FunAudioLLM/SenseVoiceSmall
      api_base: http://localhost:8000/v1
      api_key: dummy-key
```

Start the proxy with that config:

```bash
litellm --config ./config.yaml
```

Then call LiteLLM's OpenAI-compatible transcription endpoint:

```bash
curl -sS http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-1234" \
  -F model=funasr-sensevoice \
  -F file=@sample.wav
```

LiteLLM forwards the multipart request to:

```text
http://localhost:8000/v1/audio/transcriptions
```

and returns the FunASR server response to the client.

## Other FunASR models

`FunAudioLLM/SenseVoiceSmall` is a good default for Mandarin, Cantonese, English,
Japanese, and Korean. For Fun-ASR-Nano or Fun-ASR-MLT-Nano deployments, use the
model id your FunASR-compatible server accepts, for example:

```yaml
model_list:
  - model_name: fun-asr-nano
    litellm_params:
      model: custom_openai/FunAudioLLM/Fun-ASR-Nano-2512
      api_base: http://localhost:8000/v1
      api_key: dummy-key
```

FunASR is the toolkit/runtime. Exact language coverage, timestamps, hotwords,
speaker labels, rich transcription tags, and diarization behavior depend on the
selected model and server configuration.

## Troubleshooting

- Use `api_base: http://host:8000/v1`; do not include `/audio/transcriptions`.
- Use a placeholder `api_key` if your local FunASR server does not require auth.
- If the upstream server validates model ids, set `model:
  custom_openai/<exact-model-id>`.
- Test the FunASR server directly with `curl http://host:8000/v1/audio/transcriptions`
  before testing through the LiteLLM proxy.
