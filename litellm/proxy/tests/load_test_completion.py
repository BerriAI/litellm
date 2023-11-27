import time, asyncio
from openai import AsyncOpenAI
import uuid


litellm_client = AsyncOpenAI(
    api_key="test",
    base_url="http://0.0.0.0:8000"
)

async def litellm_completion():
  return await litellm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}],
        )


async def main():
    start = time.time()
    n = 1  # Number of concurrent tasks
    tasks = [litellm_completion() for _ in range(n)]
    chat_completions = await asyncio.gather(*tasks)
    successful_completions = [c for c in chat_completions if c is not None]
    print(n, time.time() - start, len(successful_completions))

if __name__ == "__main__":
    asyncio.run(main())