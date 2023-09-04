# Google Palm/VertexAI
LiteLLM supports chat-bison, chat-bison@001, text-bison, text-bison@001

### Google VertexAI Models
Sample notebook for calling VertexAI models: https://github.com/BerriAI/litellm/blob/main/cookbook/liteLLM_VertextAI_Example.ipynb

All calls using Vertex AI require the following parameters:
* Your Project ID
`litellm.vertex_project` = "hardy-device-38811" Your Project ID
* Your Project Location
`litellm.vertex_location` = "us-central1" 

Authentication:
VertexAI uses Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information on setting this up

VertexAI requires you to set `application_default_credentials.json`, this can be set by running `gcloud auth application-default login` in your terminal

| Model Name       | Function Call                                            |
|------------------|----------------------------------------------------------|
| chat-bison       | `completion('chat-bison', messages)`                    |
| chat-bison@001   | `completion('chat-bison@001', messages)`                |
| text-bison       | `completion('text-bison', messages)`                    |
| text-bison@001   | `completion('text-bison@001', messages)`                |

