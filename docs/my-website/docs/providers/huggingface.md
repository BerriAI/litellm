import Image from '@theme/IdealImage';

# Huggingface

LiteLLM supports Huggingface Inference Endpoints. It uses the [text-generation-inference](https://github.com/huggingface/text-generation-inference) format. You can use any chat/text model from Hugging Face with the following steps:

* Copy your model id/url from Huggingface Inference Endpoints
    - [ ] Go to https://ui.endpoints.huggingface.co/
    - [ ] Copy the url of the specific model you'd like to use 
    <Image img={require('../../img/hf_inference_endpoint.png')} alt="HF_Dashboard" style={{ maxWidth: '50%', height: 'auto' }}/>
* Set it as your model name
* Set your HUGGINGFACE_API_KEY as an environment variable

Need help deploying a model on huggingface? [Check out this guide.](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint)

## usage 

In this case our model id is the same as the model url - `https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud`



You need to tell LiteLLM when you're calling Huggingface. You can do that in 2 ways:

* By passing in the custom llm provider as part of the model name -  
completion(model="<custom_llm_provider>/<model_id>",...). 

```
import os 
from litellm import completion 

# Set env variables
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# model = <custom_llm_provider>/<model_id>
response = completion(model="huggingface/https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud", messages=messages)

print(response)
```

* By passing in a `custom_llm_provider` argument in the completion call

```
import os 
from litellm import completion 

# Set env variables
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

response = completion(model="https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud", messages=messages, custom_llm_provider="huggingface")
# Add any assertions here to check the response
print(response)
```