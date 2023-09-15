# VertexAI / Google Palm
LiteLLM supports chat-bison, chat-bison@001, text-bison, text-bison@001

## Google VertexAI Models
Sample notebook for calling VertexAI models: https://github.com/BerriAI/litellm/blob/main/cookbook/liteLLM_VertextAI_Example.ipynb

All calls using Vertex AI require the following parameters:
* Your Project ID
`litellm.vertex_project` = "hardy-device-38811" Your Project ID
* Your Project Location
`litellm.vertex_location` = "us-central1" 

## Pre-requisites
`pip install google-cloud-aiplatform`

Authentication:
VertexAI uses Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information on setting this up

VertexAI requires you to set `application_default_credentials.json`, this can be set by running `gcloud auth application-default login` in your terminal

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
