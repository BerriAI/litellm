# Gnani Prisma REST transcription

Use `gnani/prisma-v3.0` or `gnani/prisma-v2.5` to transcribe one mono recording of up to 30 seconds. Gnani accepts WAV, MP3, OGG, FLAC, AAC and M4A. Keep the correct extension on the upload because Gnani selects its decoder from the filename. For in-memory compressed audio, pass a tuple such as `("clip.mp3", audio_bytes)`.

Set `GNANI_API_KEY` to your Gnani key or pass `api_key` to LiteLLM. To use a different provider host, set `GNANI_API_BASE` or pass `api_base`. LiteLLM appends `/stt/v3`.

## Python SDK

```python
import asyncio
from pathlib import Path

import litellm

audio = Path("lecture.wav")
result = litellm.transcription(model="gnani/prisma-v3.0", file=audio)
print(result.text)

result_v25 = litellm.transcription(model="gnani/prisma-v2.5", file=audio, language="hi")
print(result_v25.text)

async def transcribe():
    result = await litellm.atranscription(
        model="gnani/prisma-v3.0",
        file=audio,
        language="hi,en",
        format="codemix",
        keyterms=["fractions"],
    )
    print(result.text)

asyncio.run(transcribe())
```

## Proxy and OpenAI clients

Save this as `config.yaml`, set `LITELLM_MASTER_KEY` to your proxy key, and start the proxy with `litellm --config config.yaml`.

```yaml
model_list:
  - model_name: gnani-prisma-v3
    litellm_params:
      model: gnani/prisma-v3.0
      api_key: os.environ/GNANI_API_KEY
    model_info:
      mode: audio_transcription
  - model_name: gnani-prisma-v2-5
    litellm_params:
      model: gnani/prisma-v2.5
      api_key: os.environ/GNANI_API_KEY
    model_info:
      mode: audio_transcription
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

```bash
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -F model=gnani-prisma-v3 -F file=@lecture.wav
```

```python
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key=os.environ["LITELLM_MASTER_KEY"])
result = client.audio.transcriptions.create(
    model="gnani-prisma-v3",
    file=Path("lecture.wav"),
    extra_body={"format": "codemix", "keyterms": '["fractions"]'},
)
print(result.text)
```

## Parameters and response

On v3.0, omit `language` for automatic detection or supply one to four comma-separated hints such as `en`, `hi-IN` or `hi,en`. On v2.5, pass exactly one language such as `hi` or `en-IN`. Both models return JSON with a `text` field. A successful silent clip can return an empty string.

Both models accept `format=transcribe` for conventional number formatting and `format=verbatim` for spoken forms. Prisma v3.0 also accepts `codemix` for mixed scripts and `transliterate` for Latin script. Omit `format` to use the provider's default. OpenAI's separate `response_format` accepts only `json`.

Prisma v3.0 accepts up to 200 `keyterms`, a `topic` of up to 1,600 characters, `previous_text` (it uses the last 600 characters), and one to four `speaker_names`. For word boosting on v2.5, pass `bias_list=["Prisma"]` and `bias_score=1.5`. For digits in the target script on v2.5, use `format="transcribe"` and `itn_native_numerals=True`. [Prisma v2.5 model guide](https://docs.gnani.ai/api/STT/models/prisma-v2-5).

Both models support find-and-replace. Pass `enable_substitution=True` and a `substitution_map` of up to ten rules, for example `[{"patterns": ["maths"], "replacement": "mathematics"}]`. For multipart clients, send arrays as JSON strings. Gnani validates audio and context limits and returns its errors through LiteLLM.

This integration covers REST transcription. Prisma REST has no word timestamps, speaker diarization or streaming. OpenAI `prompt` and `temperature` have no equivalent here. Gnani has no published numeric rate in the model catalog. LiteLLM can report zero spend until the deployment supplies its contracted audio rate. [Gnani REST reference](https://docs.gnani.ai/api/STT/speech-to-text) and [Prisma v3.0 model guide](https://docs.gnani.ai/api/STT/models/prisma-v3).
