# benchmark_optimization.py
import time
import asyncio
from litellm import acompletion

async def benchmark_performance():
    """Benchmark completion performance"""
    iterations = 1000
    
    async def single_call():
        return await acompletion(
            model="openai/gpt-4o",
            mock_response="Test",
            messages=[{"role": "user", "content": "Hello"}]
        )
    
    start = time.perf_counter()
    tasks = [single_call() for _ in range(iterations)]
    await asyncio.gather(*tasks)
    end = time.perf_counter()
    
    return end - start, iterations

async def main():
    total_time, total_calls = await benchmark_performance()
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Total calls: {total_calls}")
    print(f"Time per call: {(total_time / total_calls) * 1000:.4f} ms")
    print(f"Calls per second: {total_calls / total_time:.2f}")

if __name__ == "__main__":
    asyncio.run(main())