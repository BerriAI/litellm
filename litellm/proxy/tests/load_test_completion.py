import time, asyncio
from openai import AsyncOpenAI
import uuid
import traceback


litellm_client = AsyncOpenAI(base_url="http://0.0.0.0:8000", api_key="any")


async def litellm_completion():
    # Your existing code for litellm_completion goes here
    try:
        response = await litellm_client.chat.completions.create(
            model="Azure OpenAI GPT-4 Canada-East (External)",
            stream=True,
            messages=[{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}],
        )
        async for chunk in response:
            print(chunk)
        return response

    except Exception as e:
        # If there's an exception, log the error message
        with open("error_log.txt", "a") as error_log:
            error_log.write(f"Error during completion: {str(e)}\n")
        pass


async def main():
    for i in range(1000000):
        start = time.time()
        n = 1000  # Number of concurrent tasks
        tasks = [litellm_completion() for _ in range(n)]

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
