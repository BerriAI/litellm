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
import openai
from openai import AsyncOpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

litellm_client = AsyncOpenAI(
    base_url="https://exampleopenaiendpoint-production.up.railway.app/",
    api_key="sk-1234",
)


# for completion
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def completion(request: Request):
    # this proxy uses the OpenAI SDK to call a fixed endpoint

    response = await litellm.acompletion(
        model="openai/anything",
        messages=[
            {
                "role": "user",
                "content": "hello who are you",
            }
        ],
        client=litellm_client,
    )

    return response


if __name__ == "__main__":
    import uvicorn

    # run this on 8090, 8091, 8092 and 8093
    uvicorn.run(app, host="0.0.0.0", port=8090)
