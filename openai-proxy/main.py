import litellm, os, traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from utils import set_callbacks
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
set_callbacks() # sets litellm callbacks for logging if they exist in the environment 
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


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completion(request: Request):
    try:
        data = await request.json()
        if "authorization" in request.headers: # if users pass LLM api keys as part of header
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

@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"


app.include_router(router)
