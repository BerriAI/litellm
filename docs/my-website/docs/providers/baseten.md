# Baseten 
LiteLLM supports any Text-Gen-Interface models on Baseten.

[Here's a tutorial on deploying a huggingface TGI model (Llama2, CodeLlama, WizardCoder, Falcon, etc.) on Baseten](https://truss.baseten.co/examples/performance/tgi-server)

### API KEYS
```python
import os 
os.environ["BASETEN_API_KEY"] = ""
```

### Baseten Models
Baseten provides infrastructure to deploy and serve ML models https://www.baseten.co/. Use liteLLM to easily call models deployed on Baseten.

Example Baseten Usage - Note: liteLLM supports all models deployed on Baseten

Usage: Pass `model=baseten/<Model ID>`

| Model Name       | Function Call                                  | Required OS Variables              |
|------------------|--------------------------------------------|------------------------------------|
| Falcon 7B        | `completion(model='baseten/qvv0xeq', messages=messages)`         | `os.environ['BASETEN_API_KEY']`     |
| Wizard LM        | `completion(model='baseten/q841o8w', messages=messages)`         | `os.environ['BASETEN_API_KEY']`     |
| MPT 7B Base      | `completion(model='baseten/31dxrj3', messages=messages)`         | `os.environ['BASETEN_API_KEY']`     |
