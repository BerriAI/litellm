import Image from '@theme/IdealImage';


`deepgram`, `fireworks ai`, `vision`, `admin ui`, `dependency upgrades`

## New Models

### **Deepgram Speech to Text**

New Speech to Text support for Deepgram models. [**Start Here**](https://docs.litellm.ai/docs/providers/deepgram)

```python
from litellm import transcription
import os 

# set api keys 
os.environ["DEEPGRAM_API_KEY"] = ""
audio_file = open("/path/to/audio.mp3", "rb")

response = transcription(model="deepgram/nova-2", file=audio_file)

print(f"response: {response}")
```

### **Fireworks AI - Vision** support for all models
LiteLLM supports document inlining for Fireworks AI models. This is useful for models that are not vision models, but still need to parse documents/images/etc.
LiteLLM will add `#transform=inline` to the url of the image_url, if the model is not a vision model [See Code](https://github.com/BerriAI/litellm/blob/1ae9d45798bdaf8450f2dfdec703369f3d2212b7/litellm/llms/fireworks_ai/chat/transformation.py#L114)


## Proxy Admin UI

- `Test Key` Tab displays `model` used in response

<Image img={require('../../img/release_notes/ui_model.png')} />

- `Test Key` Tab renders content in `.md`, `.py` (any code/markdown format)

<Image img={require('../../img/release_notes/ui_format.png')} />


## Dependency Upgrades

- (Security fix) Upgrade to `fastapi==0.115.5` https://github.com/BerriAI/litellm/pull/7447

## Bug Fixes

- Add health check support for realtime models [Here](https://docs.litellm.ai/docs/proxy/health#realtime-models)
- Health check error with audio_transcription model https://github.com/BerriAI/litellm/issues/5999







