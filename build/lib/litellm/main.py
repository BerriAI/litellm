import os, openai, cohere

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

