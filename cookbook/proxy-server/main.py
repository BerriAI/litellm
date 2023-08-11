from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import traceback
import litellm

from litellm import completion 
import os, dotenv
dotenv.load_dotenv()

######### LOGGING ###################
# log your data to slack, supabase
litellm.success_callback=["slack", "supabase"] # set .env SLACK_API_TOKEN, SLACK_API_SECRET, SLACK_API_CHANNEL, SUPABASE 

######### ERROR MONITORING ##########
# log errors to slack, sentry, supabase
litellm.failure_callback=["slack", "sentry", "supabase"] # .env SENTRY_API_URL

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'received!', 200

@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json
    try:
        # pass in data to completion function, unpack data
        response = completion(**data)
    except Exception as e:
        # call handle_error function
        return handle_error(data)
    return response, 200

@app.route('/get_models', methods=["POST"])
def get_models():
    try:
        return litellm.model_list
    except Exception as e:
        traceback.print_exc()
        response = {"error": str(e)}
    return response, 200

if __name__ == "__main__":
  from waitress import serve
  serve(app, host="0.0.0.0", port=5000, threads=500)

############### Advanced ##########################

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



############ Caching ###################################
# make a new endpoint with caching
# This Cache is built using ChromaDB
# it has two functions add_cache() and get_cache()
@app.route('/chat/completions', methods=["POST"])
def api_completion_with_cache():
    data = request.json
    try:
        cache_response = get_cache(data['messages'])
        if cache_response!=None:
            return cache_response
        # pass in data to completion function, unpack data
        response = completion(**data) 

        # add to cache 
    except Exception as e:
        # call handle_error function
        return handle_error(data)
    return response, 200

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