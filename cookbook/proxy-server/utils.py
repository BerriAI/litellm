
from litellm import completion 
import os, dotenv
import json
dotenv.load_dotenv()
############### Advanced ##########################

########### streaming ############################
def generate_responses(response):
    for chunk in response:
        yield json.dumps({"response": chunk}) + "\n"

################ ERROR HANDLING #####################
# implement model fallbacks, cooldowns, and retries
# if a model fails assume it was rate limited and let it cooldown for 60s
def handle_error(data):
    import time
    # retry completion() request with fallback models
    response = None
    start_time = time.time()
    rate_limited_models = set()
    model_expiration_times = {}
    fallback_strategy=['gpt-3.5-turbo', 'command-nightly', 'claude-2']
    while response == None and time.time() - start_time < 45: # retry for 45s
      for model in fallback_strategy:
        try:
            if model in rate_limited_models: # check if model is currently cooling down
              if model_expiration_times.get(model) and time.time() >= model_expiration_times[model]:
                  rate_limited_models.remove(model) # check if it's been 60s of cool down and remove model
              else:
                  continue # skip model
            print(f"calling model {model}")
            response = completion(**data)
            if response != None:
              return response
        except Exception as e:
          rate_limited_models.add(model)
          model_expiration_times[model] = time.time() + 60 # cool down this selected model
          pass
    return response


########### Pricing is tracked in Supabase ############



import uuid
cache_collection = None
# Add a response to the cache
def add_cache(messages, model_response):
    global cache_collection
    if cache_collection is None:
        make_collection()

    user_question = message_to_user_question(messages)

    # Add the user question and model response to the cache
    cache_collection.add(
        documents=[user_question],
        metadatas=[{"model_response": str(model_response)}],
        ids=[str(uuid.uuid4())]
    )
    return

# Retrieve a response from the cache if similarity is above the threshold
def get_cache(messages, similarity_threshold):
    try:
        global cache_collection
        if cache_collection is None:
            make_collection()

        user_question = message_to_user_question(messages)

        # Query the cache for the user question
        results = cache_collection.query(
            query_texts=[user_question],
            n_results=1
        )

        if len(results['distances'][0]) == 0:
            return None  # Cache is empty

        distance = results['distances'][0][0]
        sim = (1 - distance)

        if sim >= similarity_threshold:
            return results['metadatas'][0][0]["model_response"]  # Return cached response
        else:
            return None  # No cache hit
    except Exception as e:
        print("Error in get cache", e)
        raise e

# Initialize the cache collection
def make_collection():
    import chromadb
    global cache_collection
    client = chromadb.Client()
    cache_collection = client.create_collection("llm_responses")

# HELPER: Extract user's question from messages
def message_to_user_question(messages):
    user_question = ""
    for message in messages:
        if message['role'] == 'user':
            user_question += message["content"]
    return user_question