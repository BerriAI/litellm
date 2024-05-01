# import sys, os
# sys.path.insert(
#     0, os.path.abspath("../")
# )  # Adds the parent directory to the system path
from fastapi import FastAPI, Request, status, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import uuid
import litellm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

litellm_router = litellm.Router(
    model_list=[
        {
            "model_name": "anything",  # model alias -> loadbalance between models with same `model_name`
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "openai/anything",  # actual model name
                "api_key": "sk-1234",
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
            },
        }
    ]
)


# for completion
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def completion(request: Request):
    # this proxy uses the OpenAI SDK to call a fixed endpoint

    response = await litellm_router.acompletion(
        model="anything",
        messages=[
            {
                "role": "user",
                "content": "hello who are you",
            }
        ],
    )

    return response


if __name__ == "__main__":
    import uvicorn

    # run this on 8090, 8091, 8092 and 8093
    uvicorn.run(app, host="0.0.0.0", port=8090)
