from fastapi import FastAPI
import uvicorn
from memory_profiler import profile
import os
import litellm
from litellm import Router
from dotenv import load_dotenv
from litellm._uuid import uuid

load_dotenv()

model_list = [
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "azure/chatgpt-v-2",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_BASE"),
        },
        "tpm": 240000,
        "rpm": 1800,
    },
    {
        "model_name": "text-embedding-ada-002",
        "litellm_params": {
            "model": "azure/azure-embedding-model",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
        },
        "tpm": 100000,
        "rpm": 10000,
    },
]

litellm.set_verbose = True
litellm.cache = litellm.Cache(
    type="s3", s3_bucket_name="litellm-my-test-bucket-2", s3_region_name="us-east-1"
)
router = Router(model_list=model_list, set_verbose=True)

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI endpoint!"}


@profile
@app.post("/router_acompletion")
async def router_acompletion():
    question = f"This is a test: {uuid.uuid4()}" * 100
    resp = await router.aembedding(model="text-embedding-ada-002", input=question)
    print("embedding-resp", resp)

    response = await router.acompletion(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}]
    )
    print("completion-resp", response)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
