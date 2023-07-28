import os, openai, cohere, dotenv

# Loading env variables using dotenv
dotenv.load_dotenv()

####### COMPLETION MODELS ###################
open_ai_chat_completion_models = [
  'gpt-3.5-turbo', 
  'gpt-4'
]
open_ai_text_completion_models = [
    'text-davinci-003'
]

cohere_models = [
    'command-nightly',
]

openrouter_models = [
    'google/palm-2-codechat-bison',
    'google/palm-2-chat-bison',
    'openai/gpt-3.5-turbo',
    'openai/gpt-3.5-turbo-16k',
    'openai/gpt-4-32k',
    'anthropic/claude-2',
    'anthropic/claude-instant-v1',
    'meta-llama/llama-2-13b-chat',
    'meta-llama/llama-2-70b-chat'
]

####### EMBEDDING MODELS ###################
open_ai_embedding_models = [
    'text-embedding-ada-002'
]

#############################################


####### COMPLETION ENDPOINTS ################
#############################################
def completion(model, messages, azure=False):
  if azure == True:
    # azure configs
    openai.api_type = "azure"
    openai.api_base = os.environ.get("AZURE_API_BASE")
    openai.api_version = os.environ.get("AZURE_API_VERSION")
    openai.api_key =   os.environ.get("AZURE_API_KEY")
    response = openai.ChatCompletion.create(
      engine=model,
      messages = messages
    )
  elif "replicate" in model: 
    prompt = " ".join([message["content"] for message in messages])
    output = replicate.run(
      model,
      input={
        "prompt": prompt,
      })
    print(f"output: {output}")
    response = ""
    for item in output: 
      print(f"item: {item}")
      response += item
    new_response = {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
              "content": response,
              "role": "assistant"
          }
        }
      ]
    }
    print(f"new response: {new_response}")
    response = new_response
  elif model in cohere_models:
    cohere_key = os.environ.get("COHERE_API_KEY")
    co = cohere.Client(cohere_key)
    prompt = " ".join([message["content"] for message in messages])
    response = co.generate(  
      model=model,
      prompt = prompt
    )
    new_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": response[0],
                    "role": "assistant"
                }
            }
        ],
    }

    response = new_response

  elif model in open_ai_chat_completion_models:
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        messages = messages
    )
  elif model in open_ai_text_completion_models:
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    prompt = " ".join([message["content"] for message in messages])
    response = openai.Completion.create(
        model=model,
        prompt = prompt
    )

  elif model in openrouter_models:
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key = os.environ.get("OPENROUTER_API_KEY")

    prompt = " ".join([message["content"] for message in messages])

    response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      headers={ 
        "HTTP-Referer": os.environ.get("OR_SITE_URL"), # To identify your app
        "X-Title": os.environ.get("OR_APP_NAME") 
        },
    )
    reply = response.choices[0].message
  return response



### EMBEDDING ENDPOINTS ####################
def embedding(model, input=[], azure=False):
  if azure == True:
    # azure configs
    openai.api_type = "azure"
    openai.api_base = os.environ.get("AZURE_API_BASE")
    openai.api_version = os.environ.get("AZURE_API_VERSION")
    openai.api_key =   os.environ.get("AZURE_API_KEY")
    response = openai.Embedding.create(input=input, engine=model)
  elif model in open_ai_embedding_models:
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.Embedding.create(input=input, model=model)
  return response


#############################################
#############################################

