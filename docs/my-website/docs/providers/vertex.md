import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# VertexAI [Anthropic, Gemini, Model Garden]

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_VertextAI_Example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Pre-requisites
* `pip install google-cloud-aiplatform` (pre-installed on proxy docker image)
* Authentication: 
    * run `gcloud auth application-default login` See [Google Cloud Docs](https://cloud.google.com/docs/authentication/external/set-up-adc)
    * Alternatively you can set `GOOGLE_APPLICATION_CREDENTIALS`

    Here's how: [**Jump to Code**](#extra)

      - Create a service account on GCP
      - Export the credentials as a json
      - load the json and json.dump the json as a string
      - store the json string in your environment as `GOOGLE_APPLICATION_CREDENTIALS`

## Sample Usage
```python
import litellm
litellm.vertex_project = "hardy-device-38811" # Your Project ID
litellm.vertex_location = "us-central1"  # proj location

response = litellm.completion(model="gemini-pro", messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}])
```

## Usage with LiteLLM Proxy Server

Here's how to use Vertex AI with the LiteLLM Proxy Server

1. Modify the config.yaml 

  <Tabs>

  <TabItem value="completion_param" label="Different location per model">

  Use this when you need to set a different location for each vertex model

  ```yaml
  model_list:
    - model_name: gemini-vision
      litellm_params:
        model: vertex_ai/gemini-1.0-pro-vision-001
        vertex_project: "project-id"
        vertex_location: "us-central1"
    - model_name: gemini-vision
      litellm_params:
        model: vertex_ai/gemini-1.0-pro-vision-001
        vertex_project: "project-id2"
        vertex_location: "us-east"
  ```

  </TabItem>

  <TabItem value="litellm_param" label="One location all vertex models">

  Use this when you have one vertex location for all models

  ```yaml
  litellm_settings: 
    vertex_project: "hardy-device-38811" # Your Project ID
    vertex_location: "us-central1" # proj location

  model_list: 
    -model_name: team1-gemini-pro
    litellm_params: 
      model: gemini-pro
  ```

  </TabItem>

  </Tabs>

2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="team1-gemini-pro",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "team1-gemini-pro",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>

## Specifying Safety Settings 
In certain use-cases you may need to make calls to the models and pass [safety settigns](https://ai.google.dev/docs/safety_setting_gemini) different from the defaults. To do so, simple pass the `safety_settings` argument to `completion` or `acompletion`. For example:


<Tabs>

<TabItem value="sdk" label="SDK">

```python
response = completion(
    model="gemini/gemini-pro", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**Option 1: Set in config**
```yaml
model_list:
  - model_name: gemini-experimental
    litellm_params:
      model: vertex_ai/gemini-experimental
      vertex_project: litellm-epic
      vertex_location: us-central1
      safety_settings:
      - category: HARM_CATEGORY_HARASSMENT
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_HATE_SPEECH
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_SEXUALLY_EXPLICIT
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_DANGEROUS_CONTENT
        threshold: BLOCK_NONE
```

**Option 2: Set on call**

```python
response = client.chat.completions.create(
    model="gemini-experimental",
    messages=[
        {
            "role": "user",
            "content": "Can you write exploits?",
        }
    ],
    max_tokens=8192,
    stream=False,
    temperature=0.0,

    extra_body={
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ],
    }
)
```
</TabItem>
</Tabs>

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
## Anthropic 
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| claude-3-opus@20240229   | `completion('vertex_ai/claude-3-opus@20240229', messages)` |
| claude-3-sonnet@20240229   | `completion('vertex_ai/claude-3-sonnet@20240229', messages)` |
| claude-3-haiku@20240307   | `completion('vertex_ai/claude-3-haiku@20240307', messages)` |

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

model = "claude-3-sonnet@20240229"

vertex_ai_project = "your-vertex-project" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "your-vertex-location" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = completion(
    model="vertex_ai/" + model,
    messages=[{"role": "user", "content": "hi"}],
    temperature=0.7,
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
)
print("\nModel Response", response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: anthropic-vertex
      litellm_params:
        model: vertex_ai/claude-3-sonnet@20240229
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
    - model_name: anthropic-vertex
      litellm_params:
        model: vertex_ai/claude-3-sonnet@20240229
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-west-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "anthropic-vertex", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>

## Model Garden
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| llama2   | `completion('vertex_ai/<endpoint_id>', messages)` |

#### Using Model Garden

```python
from litellm import completion
import os

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = completion(
  model="vertex_ai/<your-endpoint-id>", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

## Gemini Pro
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| gemini-pro   | `completion('gemini-pro', messages)`, `completion('vertex_ai/gemini-pro', messages)` |

## Gemini Pro Vision
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| gemini-pro-vision   | `completion('gemini-pro-vision', messages)`, `completion('vertex_ai/gemini-pro-vision', messages)`|

## Gemini 1.5 Pro (and Vision)
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| gemini-1.5-pro   | `completion('gemini-1.5-pro', messages)`, `completion('vertex_ai/gemini-pro', messages)` |




#### Using Gemini Pro Vision

Call `gemini-pro-vision` in the same input/output format as OpenAI [`gpt-4-vision`](https://docs.litellm.ai/docs/providers/openai#openai-vision-models)

LiteLLM Supports the following image types passed in `url`
- Images with Cloud Storage URIs - gs://cloud-samples-data/generative-ai/image/boats.jpeg
- Images with direct links - https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg
- Videos with Cloud Storage URIs - https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4
- Base64 Encoded Local Images

**Example Request - image url**

<Tabs>

<TabItem value="direct" label="Images with direct links">

```python
import litellm

response = litellm.completion(
  model = "vertex_ai/gemini-pro-vision",
  messages=[
      {
          "role": "user",
          "content": [
                          {
                              "type": "text",
                              "text": "Whats in this image?"
                          },
                          {
                              "type": "image_url",
                              "image_url": {
                              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                              }
                          }
                      ]
      }
  ],
)
print(response)
```
</TabItem>

<TabItem value="base" label="Local Base64 Images">

```python
import litellm

def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "cached_logo.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)
response = litellm.completion(
    model="vertex_ai/gemini-pro-vision",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64_image
                    },
                },
            ],
        }
    ],
)
print(response)
```
</TabItem>
</Tabs>


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


## Extra

### Using `GOOGLE_APPLICATION_CREDENTIALS`
Here's the code for storing your service account credentials as `GOOGLE_APPLICATION_CREDENTIALS` environment variable:


```python
def load_vertex_ai_credentials():
  # Define the path to the vertex_key.json file
  print("loading vertex ai credentials")
  filepath = os.path.dirname(os.path.abspath(__file__))
  vertex_key_path = filepath + "/vertex_key.json"

  # Read the existing content of the file or create an empty dictionary
  try:
      with open(vertex_key_path, "r") as file:
          # Read the file content
          print("Read vertexai file path")
          content = file.read()

          # If the file is empty or not valid JSON, create an empty dictionary
          if not content or not content.strip():
              service_account_key_data = {}
          else:
              # Attempt to load the existing JSON content
              file.seek(0)
              service_account_key_data = json.load(file)
  except FileNotFoundError:
      # If the file doesn't exist, create an empty dictionary
      service_account_key_data = {}

  # Create a temporary file
  with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
      # Write the updated content to the temporary file
      json.dump(service_account_key_data, temp_file, indent=2)

  # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)
```


### Using GCP Service Account 

1. Figure out the Service Account bound to the Google Cloud Run service

<Image img={require('../../img/gcp_acc_1.png')} />

2. Get the FULL EMAIL address of the corresponding Service Account

3. Next, go to IAM & Admin > Manage Resources , select your top-level project that houses your Google Cloud Run Service

Click `Add Principal`

<Image img={require('../../img/gcp_acc_2.png')}/>

4. Specify the Service Account as the principal and Vertex AI User as the role

<Image img={require('../../img/gcp_acc_3.png')}/>

Once that's done, when you deploy the new container in the Google Cloud Run service, LiteLLM will have automatic access to all Vertex AI endpoints.


s/o @[Darien Kindlund](https://www.linkedin.com/in/kindlund/) for this tutorial







