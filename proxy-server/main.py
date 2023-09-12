from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
import traceback
import litellm

from litellm import completion 
import openai
from utils import handle_error, get_cache, add_cache
import os, dotenv
import logging
import json
dotenv.load_dotenv()

# TODO: set your keys in .env or here:
# os.environ["OPENAI_API_KEY"] = "" # set your openai key here
# see supported models / keys here: https://litellm.readthedocs.io/en/latest/supported/

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

def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"

@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json
    if data.get('stream') == "True":
        data['stream'] = True # convert to boolean
    try:
        # pass in data to completion function, unpack data
        response = completion(**data)
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return Response(data_generator(response), mimetype='text/event-stream')
    except Exception as e:
        # call handle_error function
        print(f"got error{e}")
        return handle_error(data)
    return response, 200 # non streaming responses

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
  serve(app, host="0.0.0.0", port=os.environ.get("PORT", 5000), threads=500)

############### Advanced ##########################

############ Caching ###################################
# make a new endpoint with caching
# This Cache is built using ChromaDB
# it has two functions add_cache() and get_cache()
@app.route('/chat/completions_with_cache', methods=["POST"])
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