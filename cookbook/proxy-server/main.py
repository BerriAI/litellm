from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import traceback
import litellm

from litellm import completion 
import os, dotenv
dotenv.load_dotenv()

######### LOGGING ###################
# log your data to slack, supabase
litellm.success_callback=["slack", "supabase"] # .env SLACK_API_TOKEN, SLACK_API_SECRET, SLACK_API_CHANNEL, SUPABASE 

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
        traceback.print_exc()
        response = {"error": str(e)}
    return response, 200

@app.route('/get_models', methods=["POST"])
def get_models():
    data = request.json
    try:
        return litellm.model_list
    except Exception as e:
        traceback.print_exc()
        response = {"error": str(e)}
    return response, 200

if __name__ == "__main__":
  from waitress import serve
  serve(app, host="0.0.0.0", port=5000, threads=500)




