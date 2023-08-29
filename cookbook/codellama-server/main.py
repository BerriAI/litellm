import traceback
from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
import traceback
import litellm
from util import handle_error
from litellm import completion 
import os, dotenv, time 
import json
dotenv.load_dotenv()

# TODO: set your keys in .env or here:
# os.environ["OPENAI_API_KEY"] = "" # set your openai key here
# os.environ["ANTHROPIC_API_KEY"] = "" # set your anthropic key here
# os.environ["TOGETHER_AI_API_KEY"] = "" # set your together ai key here
# see supported models / keys here: https://litellm.readthedocs.io/en/latest/supported/
######### ENVIRONMENT VARIABLES ##########
verbose = True

# litellm.caching_with_models = True # CACHING: caching_with_models Keys in the cache are messages + model. - to learn more: https://docs.litellm.ai/docs/caching/
######### PROMPT LOGGING ##########
os.environ["PROMPTLAYER_API_KEY"] = "" # set your promptlayer key here - https://promptlayer.com/

# set callbacks
litellm.success_callback = ["promptlayer"]
############ HELPER FUNCTIONS ###################################

def print_verbose(print_statement):
    if verbose:
        print(print_statement)

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
    start_time = time.time() 
    if data.get('stream') == "True":
        data['stream'] = True # convert to boolean
    try:
        if "prompt" not in data:
            raise ValueError("data needs to have prompt")
        data["model"] = "togethercomputer/CodeLlama-34b-Instruct" # by default use Together AI's CodeLlama model - https://api.together.xyz/playground/chat?model=togethercomputer%2FCodeLlama-34b-Instruct
        # COMPLETION CALL
        system_prompt = "Only respond to questions about code. Say 'I don't know' to anything outside of that."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": data.pop("prompt")}]
        data["messages"] = messages
        print(f"data: {data}")
        response = completion(**data)
        ## LOG SUCCESS
        end_time = time.time() 
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return Response(data_generator(response), mimetype='text/event-stream')
    except Exception as e:
        # call handle_error function
        print_verbose(f"Got Error api_completion(): {traceback.format_exc()}")
        ## LOG FAILURE
        end_time = time.time() 
        traceback_exception = traceback.format_exc()
        return handle_error(data=data)
    return response

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
  serve(app, host="0.0.0.0", port=4000, threads=500)

