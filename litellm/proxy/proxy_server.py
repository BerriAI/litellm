import litellm
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
user_api_base = None
user_model = None

def initialize(model, api_base):
    global user_model, user_api_base
    user_model = model
    user_api_base = api_base


# for streaming
def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"
        
@app.get("/models") # if project requires model list 
def model_list(): 
    return dict(
        data=[{"id": user_model, "object": "model", "created": 1677610602, "owned_by": "openai"}],
        object="list",
    )

@app.post("/chat/completions")
async def completion(request: Request):
    data = await request.json()
    if (user_model is None):
        raise ValueError("Proxy model needs to be set")
    data["model"] = user_model
    if user_api_base:
        data["api_base"] = user_api_base
    response = litellm.completion(**data)
    if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(data_generator(response), media_type='text/event-stream')
    return response