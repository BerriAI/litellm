import litellm, os, traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from typing import Optional
try:
    from utils import set_callbacks, load_router_config
except ImportError:
    from litellm_server.utils import set_callbacks, load_router_config
import dotenv
dotenv.load_dotenv() # load env variables

app = FastAPI(docs_url="/", title="LiteLLM API")
router = APIRouter()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#### GLOBAL VARIABLES ####
llm_router: Optional[litellm.Router] = None

set_callbacks() # sets litellm callbacks for logging if they exist in the environment 
llm_router = load_router_config(router=llm_router)
#### API ENDPOINTS ####
@router.post("/v1/models")
@router.get("/models")  # if project requires model list
def model_list():
    all_models = litellm.utils.get_valid_models()
    return dict(
        data=[
            {
                "id": model,
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
            }
            for model in all_models
        ],
        object="list",
    )
# for streaming
def data_generator(response):
    print("inside generator")
    for chunk in response:
        print(f"returned chunk: {chunk}")
        yield f"data: {json.dumps(chunk)}\n\n"

@router.post("/v1/completions")
@router.post("/completions")
async def completion(request: Request):
    data = await request.json()
    response = litellm.completion(
        **data
    )
    if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(data_generator(response), media_type='text/event-stream')
    return response

@router.post("/v1/embeddings")
@router.post("/embeddings")
async def embedding(request: Request):
    try: 
        data = await request.json() 
        # default to always using the "ENV" variables, only if AUTH_STRATEGY==DYNAMIC then reads headers
        if os.getenv("AUTH_STRATEGY", None) == "DYNAMIC" and "authorization" in request.headers: # if users pass LLM api keys as part of header
            api_key = request.headers.get("authorization")
            api_key = api_key.replace("Bearer", "").strip() 
            if len(api_key.strip()) > 0:
                api_key = api_key
                data["api_key"] = api_key
        response = litellm.embedding(
            **data
        )
        return response
    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        return {"error": error_msg}

@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completion(request: Request):
    try:
        data = await request.json()
        # default to always using the "ENV" variables, only if AUTH_STRATEGY==DYNAMIC then reads headers
        if os.getenv("AUTH_STRATEGY", None) == "DYNAMIC" and "authorization" in request.headers: # if users pass LLM api keys as part of header
            api_key = request.headers.get("authorization")
            api_key = api_key.replace("Bearer", "").strip() 
            if len(api_key.strip()) > 0:
                api_key = api_key
                data["api_key"] = api_key
        response = litellm.completion(
            **data
        )
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
                return StreamingResponse(data_generator(response), media_type='text/event-stream')
        return response
    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        return {"error": error_msg}
        # raise HTTPException(status_code=500, detail=error_msg)

@router.post("/router/completions")
async def router_completion(request: Request):
    global llm_router
    try: 
        data = await request.json()
        if "model_list" in data: 
            llm_router = litellm.Router(model_list=data.pop("model_list"))
        if llm_router is None: 
            raise Exception("Save model list via config.yaml. Eg.: ` docker build -t myapp --build-arg CONFIG_FILE=myconfig.yaml .` or pass it in as model_list=[..] as part of the request body")
        
        # openai.ChatCompletion.create replacement
        response = await llm_router.acompletion(model="gpt-3.5-turbo", 
                        messages=[{"role": "user", "content": "Hey, how's it going?"}])

        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
                return StreamingResponse(data_generator(response), media_type='text/event-stream')
        return response
    except Exception as e: 
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        return {"error": error_msg}

@router.post("/router/embedding")
async def router_embedding(request: Request):
    global llm_router
    try: 
        data = await request.json()
        if "model_list" in data: 
            llm_router = litellm.Router(model_list=data.pop("model_list"))
        if llm_router is None: 
            raise Exception("Save model list via config.yaml. Eg.: ` docker build -t myapp --build-arg CONFIG_FILE=myconfig.yaml .` or pass it in as model_list=[..] as part of the request body")

        response = await llm_router.aembedding(model="gpt-3.5-turbo", 
                        messages=[{"role": "user", "content": "Hey, how's it going?"}])

        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
                return StreamingResponse(data_generator(response), media_type='text/event-stream')
        return response
    except Exception as e: 
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        return {"error": error_msg}

@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"


app.include_router(router)
