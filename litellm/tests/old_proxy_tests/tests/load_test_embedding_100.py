import time, asyncio
from openai import AsyncOpenAI
import uuid
import traceback


litellm_client = AsyncOpenAI(api_key="test", base_url="http://0.0.0.0:8000")


async def litellm_completion():
    # Your existing code for litellm_completion goes here
    try:
        print("starting embedding calls")
        response = await litellm_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[
                "hello who are you" * 2000,
                "hello who are you tomorrow 1234" * 1000,
                "hello who are you tomorrow 1234" * 1000,
            ],
        )
        print(response)
        return response

    except Exception as e:
        # If there's an exception, log the error message
        with open("error_log.txt", "a") as error_log:
            error_log.write(f"Error during completion: {str(e)}\n")
        pass


async def main():
    start = time.time()
    n = 100  # Number of concurrent tasks
    tasks = [litellm_completion() for _ in range(n)]

    chat_completions = await asyncio.gather(*tasks)

    successful_completions = [c for c in chat_completions if c is not None]

    # Write errors to error_log.txt
    with open("error_log.txt", "a") as error_log:
        for completion in chat_completions:
            if isinstance(completion, str):
                error_log.write(completion + "\n")

    print(n, time.time() - start, len(successful_completions))


if __name__ == "__main__":
    # Blank out contents of error_log.txt
    open("error_log.txt", "w").close()

    asyncio.run(main())
