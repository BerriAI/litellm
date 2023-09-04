# Together AI 
LiteLLM supports all models on Together AI. 

### API KEYS

```python 
import os 
os.environ["TOGETHERAI_API_KEY"] = ""
```

### Together AI Models
liteLLM supports `non-streaming` and `streaming` requests to all models on https://api.together.xyz/

Example TogetherAI Usage - Note: liteLLM supports all models deployed on TogetherAI

| Model Name                        | Function Call                                                          | Required OS Variables           |
|-----------------------------------|------------------------------------------------------------------------|---------------------------------|
| togethercomputer/llama-2-70b-chat  | `completion('togethercomputer/llama-2-70b-chat', messages)`            | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/LLaMA-2-13b-chat  | `completion('togethercomputer/LLaMA-2-13b-chat', messages)`            | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/code-and-talk-v1 | `completion('togethercomputer/code-and-talk-v1', messages)`           | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/creative-v1      | `completion('togethercomputer/creative-v1', messages)`                | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/yourmodel        | `completion('togethercomputer/yourmodel', messages)`                  | `os.environ['TOGETHERAI_API_KEY']` |
