"""
High-concurrency load generator for litellm proxy memory leak reproduction.

Fires async requests at the proxy as fast as possible.
Run: python blast.py [--url URL] [--concurrency N] [--total N]
"""

import argparse
import asyncio
import time
import aiohttp


PAYLOAD = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Say hello."}],
}


async def worker(session, url, headers, results, worker_id):
    """Single worker coroutine — sends requests in a loop."""
    while True:
        try:
            async with session.post(url, json=PAYLOAD, headers=headers) as resp:
                await resp.read()
                if resp.status == 200:
                    results["ok"] += 1
                else:
                    results["err"] += 1
        except Exception:
            results["err"] += 1
        results["total"] += 1


async def main(proxy_url, concurrency, total_requests):
    url = f"{proxy_url}/chat/completions"
    headers = {
        "Authorization": "Bearer sk-test-master-key",
        "Content-Type": "application/json",
    }

    results = {"ok": 0, "err": 0, "total": 0}

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Start workers
        tasks = []
        for i in range(concurrency):
            tasks.append(asyncio.create_task(worker(session, url, headers, results, i)))

        t0 = time.time()
        last_report = t0
        last_total = 0

        while results["total"] < total_requests:
            await asyncio.sleep(1.0)
            now = time.time()
            elapsed = now - t0
            interval = now - last_report
            interval_count = results["total"] - last_total
            rps = interval_count / interval if interval > 0 else 0
            print(
                f"[{elapsed:6.1f}s] total={results['total']:>7d}  ok={results['ok']:>7d}  "
                f"err={results['err']:>5d}  rps={rps:>6.0f}"
            )
            last_report = now
            last_total = results["total"]

        # Cancel workers
        for t in tasks:
            t.cancel()

        elapsed = time.time() - t0
        print(f"\n=== DONE === {results['total']} requests in {elapsed:.1f}s "
              f"({results['total']/elapsed:.0f} rps)  ok={results['ok']} err={results['err']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4000")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--total", type=int, default=50000)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.concurrency, args.total))
