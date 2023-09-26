import litellm
import click, json
from dotenv import load_dotenv
load_dotenv()
try:
    from fastapi import FastAPI, Request, status, HTTPException, Depends
    from fastapi.responses import StreamingResponse
except:
    raise ImportError("FastAPI needs to be imported. Run - `pip install fastapi`")

try:
    import uvicorn
except:
    raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")

app = FastAPI()
user_api_base = None
user_model = None


# for streaming
def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"

@app.get("/models") # if project requires model list 
def model_list(): 
    return dict(
        data=[
            {"id": user_model, "object": "model", "created": 1677610602, "owned_by": "openai"}
        ],
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


@click.command()
@click.option('--port', default=8000, help='Port to bind the server to.')
@click.option('--api_base',default=None, help='API base URL.')
@click.option('--model', required=True, help='The model name to pass to litellm expects') 
def run_server(port, api_base, model):
    global user_api_base, user_model
    user_api_base = api_base
    user_model = model
    uvicorn.run(app, host='0.0.0.0', port=port)