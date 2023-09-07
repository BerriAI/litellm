# Aleph Alpha

LiteLLM supports all models from [Aleph Alpha](https://www.aleph-alpha.com/). 

Like AI21 and Cohere, you can use these models without a waitlist. 

### API KEYS
```python
import os
os.environ["ALEPHALPHA_API_KEY"] = ""
```

### Aleph Alpha Models
https://www.aleph-alpha.com/

| Model Name       | Function Call                                  | Required OS Variables              |
|------------------|--------------------------------------------|------------------------------------|
| luminous-base       | `completion(model='luminous-base', messages=messages)`         | `os.environ['ALEPHALPHA_API_KEY']`     |
| luminous-base-control       | `completion(model='luminous-base-control', messages=messages)`         | `os.environ['ALEPHALPHA_API_KEY']`     |
| luminous-extended       | `completion(model='luminous-extended', messages=messages)`         | `os.environ['ALEPHALPHA_API_KEY']`     |
| luminous-extended-control       | `completion(model='luminous-extended-control', messages=messages)`         | `os.environ['ALEPHALPHA_API_KEY']`     |
| luminous-supreme     | `completion(model='luminous-supreme', messages=messages)`         | `os.environ['ALEPHALPHA_API_KEY']`     |
| luminous-supreme-control     | `completion(model='luminous-supreme-control', messages=messages)`         | `os.environ['ALEPHALPHA_API_KEY']`     |
