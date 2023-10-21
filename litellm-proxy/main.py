
import litellm
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()
router = APIRouter()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    litellm.set_verbose=True
    data = await request.json()

    api_key = request.headers.get("authorization")
    api_key = api_key.split(" ")[1]
    data["api_key"] = api_key
    response = litellm.completion(
        **data
    )
    if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(data_generator(response), media_type='text/event-stream')
    return response

@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"


app.include_router(router)
