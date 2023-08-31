from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
from litellm import completion 
import os, dotenv
import random
dotenv.load_dotenv()

# TODO: set your keys in .env or here:
# os.environ["OPENAI_API_KEY"] = "" # set your openai key here or in your .env 
# see supported models, keys here: 


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'received!', 200

# Dictionary of LLM functions with their A/B test ratios, should sum to 1 :) 
llm_dict = {
    "gpt-4": 0.2,
    "together_ai/togethercomputer/llama-2-70b-chat": 0.4,
    "claude-2": 0.2,
    "claude-1.2": 0.2
}


@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json
    try:
        # pass in data to completion function, unpack data
        selected_llm = random.choices(list(llm_dict.keys()), weights=list(llm_dict.values()))[0]
        response = completion(**data, model=selected_llm)
    except Exception as e:
        print(f"got error{e}")
    return response, 200


if __name__ == "__main__":
  from waitress import serve
  print("starting server")
  serve(app, host="0.0.0.0", port=5000, threads=500)

