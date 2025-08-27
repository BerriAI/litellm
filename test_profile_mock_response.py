from line_profiler import profile
import asyncio
import time

@profile
def test_mock_response():
    from litellm import acompletion
    from litellm.main import reset_performance_timings, print_performance_summary
    
    # Reset timing data before starting
    reset_performance_timings()
    
    async def make_request():
        response = await acompletion(
            model="openai/gpt-4o",
            mock_response="Hello, world!",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        return response
    
    async def run_concurrent_batch():
        # Create 1000 concurrent requests
        tasks = [make_request() for _ in range(2000)]
        results = await asyncio.gather(*tasks)
        return results
    
    async def run_test():
        # Run 1000 concurrent requests every second for 5 seconds
        all_tasks = []
        
        for second in range(5):
            print(f"Starting batch {second + 1}/5 at {time.time()}")
            task = asyncio.create_task(run_concurrent_batch())
            all_tasks.append(task)
            
            # Wait 1 second before starting next batch (except for the last one)
            if second < 4:
                await asyncio.sleep(1)
        
        # Wait for all batches to complete
        print("Waiting for all requests to complete...")
        all_results = await asyncio.gather(*all_tasks)
        print(f"Completed all 5000 requests")
        # Print performance timing summary
        print_performance_summary()
        return all_results
    
    # Run the async test
    results = asyncio.run(run_test())
    

    return results

if __name__ == "__main__":
    test_mock_response()
    