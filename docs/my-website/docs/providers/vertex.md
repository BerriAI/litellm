# VertexAI - Google

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_VertextAI_Example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Pre-requisites
* `pip install google-cloud-aiplatform`
* Authentication: 
    * run `gcloud auth application-default login` See [Google Cloud Docs](https://cloud.google.com/docs/authentication/external/set-up-adc)
    * Alternatively you can set `application_default_credentials.json`

## Set Vertex Project & Vertex Location
All calls using Vertex AI require the following parameters:
* Your Project ID
```python
import os, litellm 

# set via env var
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811" # Your Project ID`

### OR ###

# set directly on module 
litellm.vertex_project = "hardy-device-38811" # Your Project ID`
```
* Your Project Location
```python
import os, litellm 

# set via env var
os.environ["VERTEXAI_LOCATION"] = "us-central1 # Your Location

### OR ###

# set directly on module 
litellm.vertex_location = "us-central1 # Your Location
```

## Sample Usage
```python
import litellm
litellm.vertex_project = "hardy-device-38811" # Your Project ID
litellm.vertex_location = "us-central1"  # proj location

response = completion(model="chat-bison", messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}])
```

## Chat Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| chat-bison-32k   | `completion('chat-bison-32k', messages)` |
| chat-bison       | `completion('chat-bison', messages)`     |
| chat-bison@001   | `completion('chat-bison@001', messages)` |

## Code Chat Models
| Model Name           | Function Call                              |
|----------------------|--------------------------------------------|
| codechat-bison       | `completion('codechat-bison', messages)`     |
| codechat-bison-32k   | `completion('codechat-bison-32k', messages)` |
| codechat-bison@001   | `completion('codechat-bison@001', messages)` |

## Text Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| text-bison       | `completion('text-bison', messages)` |
| text-bison@001   | `completion('text-bison@001', messages)` |

## Code Text Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| code-bison       | `completion('code-bison', messages)` |
| code-bison@001   | `completion('code-bison@001', messages)` |
| code-gecko@001   | `completion('code-gecko@001', messages)` |
| code-gecko@latest| `completion('code-gecko@latest', messages)` |
