import sys, os, platform
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

try:
    import uvicorn
    import fastapi
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn", "fastapi"])
print()
print("\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m")
print()
print("\033[1;34mDocs: https://docs.litellm.ai/docs/proxy_server\033[0m")
print() 

import litellm
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
import json
import logging

app = FastAPI()
router = APIRouter()

user_api_base = None
user_model = None
user_debug = False
user_max_tokens = None
user_temperature = None
user_telemetry = False

#### HELPER FUNCTIONS ####
def print_verbose(print_statement):
    global user_debug 
    if user_debug: 
         print(print_statement)

def usage_telemetry(): # helps us know if people are using this feature. Set `litellm --telemetry False` to your cli call to turn this off
    if user_telemetry: 
        data = {
            "feature": "local_proxy_server"
        }
        litellm.utils.litellm_telemetry(data=data)

def initialize(model, api_base, debug, temperature, max_tokens, max_budget, telemetry, drop_params, add_function_to_prompt):
    global user_model, user_api_base, user_debug, user_max_tokens, user_temperature, user_telemetry
    user_model = model
    user_api_base = api_base
    user_debug = debug
    user_max_tokens = max_tokens
    user_temperature = temperature
    user_telemetry = telemetry
    usage_telemetry()
    if drop_params == True: 
        litellm.drop_params = True
    if add_function_to_prompt == True: 
        litellm.add_function_to_prompt = True
    if max_budget: 
        litellm.max_budget = max_budget

def deploy_proxy(model, api_base, debug, temperature, max_tokens, telemetry, deploy):
    import requests
    # Load .env file

    # Prepare data for posting
    data = {
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # print(data)

    # Make post request to the url
    url = "https://litellm-api.onrender.com/deploy"
    # url = "http://0.0.0.0:4000/deploy"

    with open(".env", "w") as env_file:
        for row in data:
            env_file.write(f"{row.upper()}='{data[row]}'\n")
        env_file.write("\n\n")
        for key in os.environ:
            value = os.environ[key]
            env_file.write(f"{key}='{value}'\n")
        # env_file.write(str(os.environ))

    files = {"file": open(".env", "rb")}
    # print(files)



    response = requests.post(url, data=data, files=files)
    # print(response)
    # Check the status of the request
    if response.status_code != 200:
        return f"Request to url: {url} failed with status: {response.status_code}"

    # Reading the response
    response_data = response.json()
    # print(response_data)
    url = response_data["url"]
    # # Do something with response_data

    return url


# for streaming
def data_generator(response):
    print("inside generator")
    for chunk in response:
        print(f"chunk: {chunk}")
        print_verbose(f"returned chunk: {chunk}")
        yield f"data: {json.dumps(chunk)}\n\n"

def litellm_completion(data, type): 
    try: 
        if user_model:
            data["model"] = user_model
        # override with user settings
        if user_temperature: 
            data["temperature"] = user_temperature
        if user_max_tokens: 
            data["max_tokens"] = user_max_tokens
        if user_api_base: 
            data["api_base"] = user_api_base
        ## CUSTOM PROMPT TEMPLATE ##  - run `litellm --config` to set this
        litellm.register_prompt_template(
            model=user_model, 
            roles={
                "system": {
                    "pre_message": os.getenv("MODEL_SYSTEM_MESSAGE_START_TOKEN", ""),
                    "post_message": os.getenv("MODEL_SYSTEM_MESSAGE_END_TOKEN", ""),    
                }, 
                "assistant": {
                    "pre_message": os.getenv("MODEL_ASSISTANT_MESSAGE_START_TOKEN", ""), 
                    "post_message": os.getenv("MODEL_ASSISTANT_MESSAGE_END_TOKEN", "")
                }, 
                "user": {
                    "pre_message": os.getenv("MODEL_USER_MESSAGE_START_TOKEN", ""), 
                    "post_message": os.getenv("MODEL_USER_MESSAGE_END_TOKEN", "")
                }
            },
            initial_prompt_value=os.getenv("MODEL_PRE_PROMPT", ""), 
            final_prompt_value=os.getenv("MODEL_POST_PROMPT", "")
        )
        if type == "completion": 
            response = litellm.text_completion(**data)
        elif type == "chat_completion": 
            response = litellm.completion(**data)
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(data_generator(response), media_type='text/event-stream')
        print_verbose(f"response: {response}")
        return response
    except Exception as e: 
        if "Invalid response object from API" in str(e): 
            completion_call_details = {}
            if user_model: 
                completion_call_details["model"] = user_model
            else: 
                completion_call_details["model"] = data['model']
            
            if user_api_base: 
                completion_call_details["api_base"] = user_api_base
            else: 
                completion_call_details["api_base"] = None
            print(f"\033[1;31mLiteLLM.Exception: Invalid API Call. Call details: Model: \033[1;37m{completion_call_details['model']}\033[1;31m; LLM Provider: \033[1;37m{e.llm_provider}\033[1;31m; Custom API Base - \033[1;37m{completion_call_details['api_base']}\033[1;31m\033[0m")
            if completion_call_details["api_base"] == "http://localhost:11434": 
                print()
                print("Trying to call ollama? Try `litellm --model ollama/llama2 --api_base http://localhost:11434`")
                print()
        else: 
            print(f"\033[1;31mLiteLLM.Exception: {str(e)}\033[0m")
        return {"message": "An error occurred"}, 500

#### API ENDPOINTS ####
@router.get("/models") # if project requires model list 
def model_list():
    if user_model != None:
        return dict(
            data=[{"id": user_model, "object": "model", "created": 1677610602, "owned_by": "openai"}],
            object="list",
        )
    else:
        all_models = litellm.model_list
        return dict(
            data = [{"id": model, "object": "model", "created": 1677610602, "owned_by": "openai"} for model in all_models],
            object="list",
        )

@router.post("/completions")
async def completion(request: Request):
    data = await request.json()
    return litellm_completion(data=data, type="completion")

@router.post("/chat/completions")
async def chat_completion(request: Request):
    data = await request.json()
    print_verbose(f"data passed in: {data}")
    response = litellm_completion(data, type="chat_completion")
    # track cost of this response, using litellm.completion_cost
    track_cost(response)
    return response

async def track_cost(response):
    try:
        logging.basicConfig(
            filename='cost.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        import datetime

        response_cost = litellm.completion_cost(completion_response=response)

        logging.info(f"Model {response.model} Cost: {response_cost:.8f}")

    except:
        pass



@router.get("/ollama_logs")
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser('~/.ollama/logs/server.log')
    return FileResponse(filepath)

app.include_router(router)