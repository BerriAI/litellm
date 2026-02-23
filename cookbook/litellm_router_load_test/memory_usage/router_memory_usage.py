#### What this tests ####

from memory_profiler import profile
import sys
import os
import time
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from dotenv import load_dotenv
from litellm._uuid import uuid

load_dotenv()


model_list = [
    {
        "model_name": "gpt-3.5-turbo",  # openai model name
        "litellm_params": {  # params for litellm completion/embedding call
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
            "api_key": os.environ["AZURE_API_KEY"],
            "api_base": os.environ["AZURE_API_BASE"],
        },
        "tpm": 100000,
        "rpm": 10000,
    },
]
litellm.set_verbose = True
litellm.cache = litellm.Cache(
    type="s3", s3_bucket_name="litellm-my-test-bucket-2", s3_region_name="us-east-1"
)
router = Router(
    model_list=model_list,
    set_verbose=True,
)  # type: ignore


@profile
async def router_acompletion():
    # embedding call
    question = f"This is a test: {uuid.uuid4()}" * 100
    resp = await router.aembedding(model="text-embedding-ada-002", input=question)
    print("embedding-resp", resp)

    response = await router.acompletion(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}]
    )
    print("completion-resp", response)
    return response


async def main():
    for i in range(1):
        start = time.time()
        n = 50  # Number of concurrent tasks
        tasks = [router_acompletion() for _ in range(n)]

        chat_completions = await asyncio.gather(*tasks)

        successful_completions = [c for c in chat_completions if c is not None]

        # Write errors to error_log.txt
        with open("error_log.txt", "a") as error_log:
            for completion in chat_completions:
                if isinstance(completion, str):
                    error_log.write(completion + "\n")

        print(n, time.time() - start, len(successful_completions))
        time.sleep(10)


if __name__ == "__main__":
    # Blank out contents of error_log.txt
    open("error_log.txt", "w").close()

    asyncio.run(main())
