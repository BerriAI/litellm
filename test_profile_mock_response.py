# test_profile_mock_response.py  
from line_profiler import profile
import asyncio

@profile
def test_mock_response():
    from litellm import acompletion
    
    async def make_request():
        return await acompletion(
            model="openai/gpt-4o",
            mock_response="Hello, world!",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    async def run_test():
        # Run 2000 concurrent requests for 5 batches
        all_tasks = []
        for second in range(5):
            tasks = [make_request() for _ in range(2000)]
            batch_task = asyncio.create_task(asyncio.gather(*tasks))
            all_tasks.append(batch_task)
            if second < 4:
                await asyncio.sleep(1)
        
        results = await asyncio.gather(*all_tasks)
        return results

    return asyncio.run(run_test())

if __name__ == "__main__":
    test_mock_response()