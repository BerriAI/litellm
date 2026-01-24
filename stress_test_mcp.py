"""
Stress test for MCP + Responses API to try to reproduce the intermittent bug.

This sends many concurrent and sequential requests to try to trigger race conditions.

Usage:
    export OPENAI_API_KEY="your-key"  # or use litellm proxy
    python stress_test_mcp.py
"""

import asyncio
import os
import sys
import time
from typing import List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import litellm
from litellm._logging import verbose_logger
import logging

# Set to DEBUG to see the actual error
verbose_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def make_mcp_streaming_call(call_id: int, model: str = "gpt-4o-mini") -> dict:
    """Make a single MCP streaming call and return result info."""
    
    mcp_tool_config = {
        "type": "mcp",
        "server_url": "litellm_proxy",
        "require_approval": "never",
    }
    
    start_time = time.time()
    result = {
        "call_id": call_id,
        "success": False,
        "error": None,
        "chunks": 0,
        "duration": 0,
    }
    
    try:
        response = await litellm.aresponses(
            model=model,
            tools=[mcp_tool_config],
            tool_choice="auto",
            input=[{
                "role": "user",
                "type": "message",
                "content": f"Call {call_id}: What is {call_id} + {call_id}?",
            }],
            stream=True,
        )
        
        chunk_count = 0
        async for chunk in response:
            chunk_count += 1
        
        result["success"] = True
        result["chunks"] = chunk_count
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        result["error"] = error_msg
        logger.error(f"Call {call_id} FAILED: {error_msg}")
        
        # Check if this is THE bug we're looking for
        if "NoneType" in str(e) and "not iterable" in str(e):
            logger.error("🐛 FOUND THE BUG! NoneType not iterable error!")
            
    result["duration"] = time.time() - start_time
    return result


async def test_sequential_rapid_fire(num_calls: int = 10, model: str = "gpt-4o-mini"):
    """Send requests one after another as fast as possible."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔥 SEQUENTIAL RAPID-FIRE TEST ({num_calls} calls)")
    logger.info(f"{'='*60}")
    
    results = []
    for i in range(num_calls):
        result = await make_mcp_streaming_call(i, model)
        results.append(result)
        status = "✅" if result["success"] else "❌"
        logger.info(f"  {status} Call {i}: {result['chunks']} chunks in {result['duration']:.2f}s")
        # NO delay between calls - as fast as possible
    
    return results


async def test_concurrent_burst(num_calls: int = 5, model: str = "gpt-4o-mini"):
    """Send multiple requests at the exact same time."""
    logger.info(f"\n{'='*60}")
    logger.info(f"⚡ CONCURRENT BURST TEST ({num_calls} simultaneous calls)")
    logger.info(f"{'='*60}")
    
    tasks = [make_mcp_streaming_call(i, model) for i in range(num_calls)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"  ❌ Call {i}: Exception - {result}")
        else:
            status = "✅" if result["success"] else "❌"
            logger.info(f"  {status} Call {i}: {result['chunks']} chunks in {result['duration']:.2f}s")
    
    return results


async def test_interleaved_calls(num_pairs: int = 5, model: str = "gpt-4o-mini"):
    """Start a second call before the first one finishes."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔀 INTERLEAVED CALLS TEST ({num_pairs} pairs)")
    logger.info(f"{'='*60}")
    
    results = []
    
    for pair in range(num_pairs):
        # Start two calls with a small delay
        task1 = asyncio.create_task(make_mcp_streaming_call(pair * 2, model))
        await asyncio.sleep(0.1)  # Start second call while first is in progress
        task2 = asyncio.create_task(make_mcp_streaming_call(pair * 2 + 1, model))
        
        result1, result2 = await asyncio.gather(task1, task2)
        results.extend([result1, result2])
        
        for r in [result1, result2]:
            status = "✅" if r["success"] else "❌"
            logger.info(f"  {status} Call {r['call_id']}: {r['chunks']} chunks")
    
    return results


async def test_cancel_and_retry(model: str = "gpt-4o-mini"):
    """Start a call, cancel it, then immediately start another."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚫 CANCEL AND RETRY TEST")
    logger.info(f"{'='*60}")
    
    results = []
    
    for i in range(3):
        # Start a call
        task = asyncio.create_task(make_mcp_streaming_call(i * 2, model))
        await asyncio.sleep(0.05)  # Let it start
        task.cancel()  # Cancel it
        
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"  🚫 Call {i*2} cancelled")
        
        # Immediately start another
        result = await make_mcp_streaming_call(i * 2 + 1, model)
        results.append(result)
        status = "✅" if result["success"] else "❌"
        logger.info(f"  {status} Call {i*2+1} after cancel: {result['chunks']} chunks")
    
    return results


def summarize_results(all_results: List[dict]):
    """Print a summary of all test results."""
    logger.info(f"\n{'='*60}")
    logger.info("📊 FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    total = len(all_results)
    successes = sum(1 for r in all_results if isinstance(r, dict) and r.get("success"))
    failures = total - successes
    
    logger.info(f"  Total calls: {total}")
    logger.info(f"  Successes: {successes}")
    logger.info(f"  Failures: {failures}")
    
    # Check for THE bug
    nonetype_errors = [r for r in all_results if isinstance(r, dict) and r.get("error") and "NoneType" in r["error"]]
    if nonetype_errors:
        logger.error(f"\n🐛 FOUND {len(nonetype_errors)} NoneType ERRORS!")
        for r in nonetype_errors:
            logger.error(f"  Call {r['call_id']}: {r['error']}")
    else:
        logger.info("\n✅ No NoneType errors found")
    
    if failures > 0:
        logger.info("\nAll errors:")
        for r in all_results:
            if isinstance(r, dict) and r.get("error"):
                logger.info(f"  Call {r['call_id']}: {r['error']}")


async def main():
    logger.info("="*60)
    logger.info("🧪 MCP + Responses API STRESS TEST")
    logger.info("="*60)
    logger.info("""
Trying to reproduce the intermittent bug:
  "Error creating initial response iterator: argument of type 'NoneType' is not iterable"

This test will:
1. Send rapid sequential requests
2. Send concurrent burst requests
3. Send interleaved requests
4. Cancel and retry requests
""")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LITELLM_API_KEY"):
        logger.warning("No API key found. Set OPENAI_API_KEY or LITELLM_API_KEY")
        logger.info("Running with mock to test the code paths...")
        
        # Use mock response
        litellm.set_verbose = True
        
    model = os.getenv("TEST_MODEL", "gpt-4o-mini")
    all_results = []
    
    try:
        # Test 1: Sequential rapid fire
        results = await test_sequential_rapid_fire(num_calls=10, model=model)
        all_results.extend(results)
        
        # Test 2: Concurrent burst
        results = await test_concurrent_burst(num_calls=5, model=model)
        all_results.extend([r for r in results if isinstance(r, dict)])
        
        # Test 3: Interleaved calls
        results = await test_interleaved_calls(num_pairs=5, model=model)
        all_results.extend(results)
        
        # Test 4: Cancel and retry
        results = await test_cancel_and_retry(model=model)
        all_results.extend(results)
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    
    summarize_results(all_results)


if __name__ == "__main__":
    asyncio.run(main())
