"""
Profile the LiteLLM proxy request path to find where time goes.
Sends N sequential requests and measures server-side timing via headers.
"""
import time
import httpx
import statistics

URL = "http://localhost:4000/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-1234"}
DATA = {"model": "fake-openai-endpoint", "max_tokens": 10, "messages": [{"role": "user", "content": "Hello"}]}
N = 200

client = httpx.Client(http2=False)

latencies = []
for i in range(N):
    t0 = time.perf_counter()
    resp = client.post(URL, json=DATA, headers=HEADERS)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies.sort()
print(f"Sequential requests: {N}")
print(f"  Mean:   {statistics.mean(latencies):.2f} ms")
print(f"  Median: {statistics.median(latencies):.2f} ms")
print(f"  P90:    {latencies[int(N*0.9)]:.2f} ms")
print(f"  P99:    {latencies[int(N*0.99)]:.2f} ms")
print(f"  Min:    {min(latencies):.2f} ms")
print(f"  Max:    {max(latencies):.2f} ms")
print(f"  Throughput: {N / sum(latencies) * 1000:.1f} req/s (sequential)")
client.close()
