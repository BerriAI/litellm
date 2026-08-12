import pytest

from litellm.llms.gandr.text_to_speech.transformation import (
    GandrTextToSpeechConfig,
)


def test_gandr_default_url():
    config = GandrTextToSpeechConfig()

    url = config.get_complete_url(
        model="gandr/mia",
        api_base=None,
        litellm_params={},
    )

    assert url == "https://tts.gandr.ai/v1/audio/speech"


def test_gandr_custom_api_base():
    config = GandrTextToSpeechConfig()

    url = config.get_complete_url(
        model="gandr/mia",
        api_base="https://tts.gandr.ai/v1",
        litellm_params={},
    )

    assert url == "https://tts.gandr.ai/v1/audio/speech"


def test_gandr_supported_openai_params():
    config = GandrTextToSpeechConfig()

    assert config.get_supported_openai_params(model="gandr/mia") == [
        "voice",
        "response_format",
        "speed",
    ]


def test_gandr_map_openai_params_default_format():
    config = GandrTextToSpeechConfig()

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"speed": 1.1},
        voice="alloy",
    )

    assert voice == "alloy"
    assert mapped["response_format"] == "wav"
    assert mapped["speed"] == 1.1


def test_gandr_map_openai_params_passthrough():
    config = GandrTextToSpeechConfig()

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"response_format": "pcm", "speed": 0.8},
        voice="gandr-dane",
    )

    assert voice == "gandr-dane"
    assert mapped["response_format"] == "pcm"
    assert mapped["speed"] == 0.8


def test_gandr_map_openai_params_requires_voice():
    config = GandrTextToSpeechConfig()

    with pytest.raises(ValueError, match="Gandr voice is required"):
        config.map_openai_params(
            model="gandr/mia",
            optional_params={},
            voice=None,
        )


def test_gandr_transform_request_body():
    config = GandrTextToSpeechConfig()

    data = config.transform_text_to_speech_request(
        model="tts-1",
        input="Hello from Gandr",
        voice="alloy",
        optional_params={"response_format": "wav", "speed": 1.0},
        litellm_params={},
        headers={},
    )

    assert data["dict_body"] == {
        "input": "Hello from Gandr",
        "model": "tts-1",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert data["headers"]["Content-Type"] == "application/json"

---

## Registration diffs (each is a one-line insertion, mirroring the elevenlabs registration exactly)

**1. `litellm/types/utils.py` - `LlmProviders` enum** (after `ELEVENLABS = "elevenlabs"`):

    GANDR = "gandr"

**2. `litellm/utils.py` - `ProviderConfigManager.get_provider_text_to_speech_config`** (insert after the `ELEVENLABS` branch):

        elif litellm.LlmProviders.GANDR == provider:
            from litellm.llms.gandr.text_to_speech.transformation import (
                GandrTextToSpeechConfig,
            )

            return GandrTextToSpeechConfig()

**3. `litellm/main.py` - `speech()` dispatch** (insert after the `custom_llm_provider == "elevenlabs"` branch):

    elif custom_llm_provider == "gandr":
        from litellm.llms.gandr.text_to_speech.transformation import (
            GandrTextToSpeechConfig,
        )

        if text_to_speech_provider_config is None:
            text_to_speech_provider_config = GandrTextToSpeechConfig()

        gandr_config: Final = cast(GandrTextToSpeechConfig, text_to_speech_provider_config)

        voice_id = voice if isinstance(voice, str) else None
        if voice_id is None or not voice_id.strip():
            raise litellm.BadRequestError(
                message="'voice' must resolve to a Gandr voice id for Gandr TTS",
                model=model,
                llm_provider=custom_llm_provider,
            )
        voice_id = voice_id.strip()

        if api_base is not None:
            litellm_params_dict["api_base"] = api_base
        if api_key is not None:
            litellm_params_dict["api_key"] = api_key

        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_id,
            text_to_speech_provider_config=gandr_config,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
            client=client,
            _is_async=aspeech or False,
        )

**4. `litellm/__init__.py`** - four lines:

gandr_models: Set = set()                                      # after elevenlabs_models: Set = set()
        elif value.get("litellm_provider") == "gandr":          # in the model-info loop
            gandr_models.add(key)
    | gandr_models                                               # in the model_list_set union
        "gandr": gandr_models,                                   # in the model_name-to-set mapping

**5. `litellm/litellm_core_utils/get_llm_provider_logic.py`** - resolve `gandr/<any>` into `gandr` for chat/completion reuse (as a non-openai-compatible-provider branch; insert before the `== "*"` fallback):

        elif model in litellm.gandr_models:
            custom_llm_provider = "gandr"

**6. `model_prices_and_context_window.json`** - one entry placed alphabetically before the `groq/*` block:

  "gandr/gandr": {
    "input_cost_per_character": 0.00005,
    "litellm_provider": "gandr",
    "metadata": {
      "calculation": "$0.05/1k characters (gandr.ai pricing, 1 credit per character)",
      "notes": "Gandr TTS, OpenAI-compatible /v1/audio/speech endpoint"
    },
    "mode": "audio_speech",
    "source": "https://gandr.ai/pricing",
    "supported_endpoints": ["/v1/audio/speech"]
  }

**7. `provider_endpoints_support.json`** - add to `"providers"`:

  "gandr": {
    "display_name": "Gandr (`gandr`)",
    "url": "https://docs.litellm.ai/docs/providers/gandr",
    "endpoints": {
      "chat_completions": true,
      "messages": true,
      "responses": true,
      "embeddings": false,
      "image_generations": false,
      "audio_transcriptions": false,
      "audio_speech": true,
      "moderations": false,
      "batches": false,
      "rerank": false,
      "a2a": false,
      "interactions": false
    }
  }

---

## Docs (PR goes to `BerriAI/litellm-docs`, not the main repo - docs moved out of main)

**File: `docs/my-website/docs/providers/gandr.md`** (sidebars: insert `"providers/gandr",` between `"providers/galadriel"` and `"providers/github"` in `sidebars.js`)


# Gandr

Gandr is a spoken-audio inference provider. Its API is OpenAI-compatible: the same `voice`, `response_format` and `speed` fields, one base URL.

| Property | Details |
|----------|---------|
| Description | Neural text-to-speech with OpenAI-compatible APIs, fast time to first audio byte |
| Provider Route on LiteLLM | `gandr/` |
| Provider Doc | [Gandr API ↗](https://gandr.ai/docs) |
| Supported Endpoints | `/audio/speech` |

## Supported Models

| Model | Route | Description |
|-------|-------|-------------|
| Gandr (default) | `gandr/gandr` | `tts-1` compatible model alias |

## Quick Start

### 1. Set the API key

```bash showLineNumbers title="Set your Gandr API key"
export GANDR_API_KEY="gnd_..."

Get a key at [https://gandr.ai](https://gandr.ai).

### 2. LiteLLM Python SDK

```python showLineNumbers title="Text-to-speech with Gandr"
import litellm

audio = litellm.speech(
    model="gandr/gandr",        # OpenAI-compatible model alias
    input="Hello from Gandr.",  # Text to synthesize
    voice="alloy",               # OpenAI voice alias or a gandr-* voice id
    api_key="gnd_...",           # optional; defaults to GANDR_API_KEY
    response_format="wav",       # wav or pcm (24 kHz, 16-bit, mono)
    speed=1.0,                   # 0.6 to 1.5
)

# audio.read() returns raw audio bytes
with open("speech.wav", "wb") as f:
    f.write(audio.read())

### 3. OpenAI Python SDK through the LiteLLM proxy

```python showLineNumbers title="Proxied OpenAI-compatible TTS"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000", api_key="sk-...")
response = client.audio.speech.create(
    model="gandr-tts",           # model alias configured in the proxy
    input="Hello from Gandr.",
    voice="alloy",
    response_format="wav",
    speed=1.0,
)
with open("speech.wav", "wb") as f:
    f.write(response.content)

## LiteLLM Proxy

### 1. Configure your proxy

```yaml showLineNumbers title="Gandr configuration in config.yaml"
model_list:
  - model_name: gandr-tts
    litellm_params:
      model: gandr/gandr
      api_key: os.environ/GANDR_API_KEY

general_settings:
  master_key: your-master-key

### 2. Make TTS requests

```bash showLineNumbers title="TTS request with curl"
curl http://localhost:4000/v1/audio/speech \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gandr-tts",
    "input": "Hello from Gandr.",
    "voice": "alloy",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav

## Supported Parameters

| Param | Type | Description |
|-------|------|-------------|
| `voice` | str | OpenAI voice alias (`alloy`, `ash`, `onyx`, `coral`, `sage`, `shimmer`, `echo`, `verse`, `ballad`, `fable`, `nova`) or a `gandr-*` voice id. Default `alloy`. |
| `response_format` | str | `wav` (default, RIFF header) or `pcm` (headerless). Anything else returns an honest 400 naming the supported formats. |
| `speed` | float | Pitch-preserving rate from 0.6 to 1.5, applied after synthesis. Out-of-range values clamp server-side. |

## Voice Aliases

LiteLLM passes the OpenAI names through to Gandr, which maps them so an unmodified client always gets audio:

| OpenAI Voice | Gandr Voice |
|--------------|-------------|
| `alloy` (default) | mia |
| `ash`, `onyx` | dane |
| `ballad`, `fable` | lewis |
| `coral`, `sage`, `shimmer` | ava |
| `echo`, `verse` | leo |
| `nova` | jenny |

Own `gandr-*` voice ids pass through unchanged.

## Common Issues

1. **Invalid API key**: Set `GANDR_API_KEY` to a valid `gnd_` token, or pass `api_key` to `litellm.speech()` for a per-call key.
2. **Unsupported format**: Gandr serves `wav` and `pcm` only (24 kHz, 16-bit, mono). No mp3 on the door; the API returns a 400 naming the supported formats.

---

## PR description

**Title:** `feat: Add Gandr text-to-speech provider`

**Body:**


## Summary

Adds Gandr as a first-class text-to-speech provider, following the repo's
existing provider shape (`litellm/llms/elevenlabs/text_to_speech`).

Gandr is an OpenAI-compatible spoken-audio inference API. Its
`POST /v1/audio/speech` endpoint accepts the OpenAI request body verbatim
(`voice`, `response_format`, `speed`) and returns raw audio bytes, so an
unmodified OpenAI SDK client works after a single `base_url` change. This
provider brings that API into LiteLLM under the `gandr/` route.

Usage:


import litellm
audio = litellm.speech(
    model="gandr/gandr",
    input="Hello from Gandr.",
    voice="alloy",
    response_format="wav",
    speed=1.0,
    api_key="gnd_...",  # or GANDR_API_KEY env var
)

## What's in the PR

- `litellm/llms/gandr/text_to_speech/transformation.py` - `GandrTextToSpeechConfig`
  extending `BaseTextToSpeechConfig`: `get_supported_openai_params`,
  `map_openai_params` (request body passthrough), `validate_environment`
  (`GANDR_API_KEY`, `x-api-key` header), `get_complete_url`
  (`https://tts.gandr.ai/v1/audio/speech`, overridable via `GANDR_API_BASE`),
  `transform_text_to_speech_request`, `transform_text_to_speech_response`
  (`HttpxBinaryResponseContent`), `get_error_class` (`GandrException`).
- `litellm/llms/gandr/common_utils.py` - `GandrException(BaseLLMException)`.
- `litellm/llms/gandr/__init__.py` and `text_to_speech/__init__.py`.
- Registration: `LlmProviders.GANDR`, `ProviderConfigManager`
  `get_provider_text_to_speech_config`, `litellm.speech()` dispatch,
  `litellm/__init__.py` model-set wiring, `get_llm_provider_logic.py`
  resolution, `model_prices_and_context_window.json` entry, and
  `provider_endpoints_support.json` (`audio_speech: true`).
- Docs: `docs/providers/gandr.md` (PR targets the litellm-docs repo).
- Tests: `tests/test_litellm/llms/gandr/test_gandr_text_to_speech_transformation.py`.

## Behavior notes

- Auth: `GANDR_API_KEY` (or per-call `api_key`) via the `x-api-key` header.
  Gandr renders at 24 kHz, 16-bit, mono; the authoritative formats are `wav`
  (RIFF header) and `pcm` (headerless).
- Voice: OpenAI voice aliases are resolved by the Gandr door (e.g. `alloy`
  maps to mia); `gandr-*` voice ids pass through unchanged.
- Speed: passed through; Gandr clamps to its real 0.6-1.5 range server-side.
- Errors surface through `GandrException`, so HTTP failures carry provider
  context the same way the other TTS providers do.

## Checklist

- [x] Provider config matches litellm provider-shape conventions
- [x] Registration wired (enum, config manager, speech dispatch, model prices)
- [x] Docs added
- [x] Tests added and passing

---

Files on disk (absolute paths):
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/litellm/llms/gandr/text_to_speech/transformation.py`
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/litellm/llms/gandr/common_utils.py`
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/litellm/llms/gandr/__init__.py`
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/litellm/llms/gandr/text_to_speech/__init__.py`
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/tests/test_litellm/llms/gandr/test_gandr_text_to_speech_transformation.py`
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/docs/gandr.md` (moves to litellm-docs `docs/providers/gandr.md`)
- `/private/tmp/claude-501/-Users-sam/6fd1390d-8eb5-4e16-81c0-b2c32436595e/scratchpad/gandr-plugin/model_prices_entry.json`

Two notes for the workflow: (1) the docs PR must target `BerriAI/litellm-docs` (the `docs/` tree was removed from the main repo); (2) a sibling subagent's LiveKit agents-js plugin and a stray `docs-gandr-tts.md` live in the same scratchpad directory, unrelated to this deliverable, left untouched.
