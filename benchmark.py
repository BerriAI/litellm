#!/usr/bin/env python3
"""
Benchmark script for LiteLLM proxy streaming performance.
Simulates the evalscope benchmark: 1000 requests at 30 req/s with streaming.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


PROMPT_TEXT = (
    "Summarize the following article - "
    "The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of "
    "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern "
    "AI were planted by philosophers who attempted to describe the process of human thinking as the "
    "mechanical manipulation of symbols. This work culminated in the invention of the programmable digital "
    "computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device "
    "and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility "
    "of building an electronic brain. The field of AI research was founded at a workshop held on the campus "
    "of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI "
    "research for decades. Many of them predicted that a machine as intelligent as a human being would exist "
    "in no more than a generation, and they were given millions of dollars to make this vision come true. "
    "Eventually, it became obvious that commercial developers and researchers had grossly underestimated the "
    "difficulty of the project. In 1974, in response to the criticism from James Lighthill and ongoing "
    "pressure from congress, the U.S. and British governments cut off all undirected, exploratory research "
    "in AI. The next few years would later be called an AI winter, a period when obtaining funding for AI "
    "projects was difficult. In the early 1980s, AI research was revived by the commercial success of expert "
    "systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By "
    "1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation "
    "computer project inspired the U.S and British governments to restore funding for academic research. "
    "However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into "
    "disrepute, and a second, longer-lasting winter began. Many researchers began to doubt that the "
    "symbolic approach would ever be able to imitate all the processes of human cognition, especially "
    "perception, robotics, learning and pattern recognition. A number of researchers began to look into "
    "sub-symbolic approaches to specific AI problems. Robotics researchers, such as Rodney Brooks, rejected "
    "symbolic AI and focused on the basic engineering problems that would allow robots to move, survive, and "
    "learn their environment. Interest in neural networks and connectionism was revived by Geoffrey Hinton, "
    "David Rumelhart and others in the middle of the 1980s. Soft computing tools were developed in the 80s, "
    "such as neural networks, fuzzy systems, Grey system theory, evolutionary computation and many tools "
    "drawn from statistics or mathematical optimization. AI gradually restored its reputation in the late "
    "1990s and early 21st century by finding specific solutions to specific problems. The narrow focus "
    "allowed researchers to produce verifiable results, exploit more mathematical methods, and collaborate "
    "with other fields (such as statistics, economics and mathematics). By 2000, solutions developed by AI "
    "researchers were being widely used, although in the 1990s they were rarely described as artificial "
    "intelligence. Faster computers, algorithmic improvements, and access to large amounts of data enabled "
    "advances in machine learning and perception; data-hungry deep learning methods started to dominate "
    "accuracy benchmarks around 2012. According to Bloomberg's Jack Clark, 2015 was a landmark year for "
    "artificial intelligence, with the number of software projects that use AI within Google increased from "
    "a 'ichever use' in 2012 to more than 2,700 projects. Clark also presents factual data indicating that "
    "error rates in image processing tasks have fallen significantly since 2011. He attributes this to an "
    "increase in affordable neural networks, due to a rise in cloud computing infrastructure and to an "
    "increase in research tools and datasets. In a 2017 survey, one in five companies reported they had "
    "incorporated AI in some offerings or processes. The amount of research into AI (measured by total "
    "publications) increased by 50% in the years 2015 through 2019. Numerous academic researchers became "
    "concerned that AI was no longer pursuing the original goal of creating versatile, fully intelligent "
    "machines. Much of current research involves statistical AI, which is overwhelmingly used to solve "
    "specific problems, even highly successful techniques such as deep learning. This concern has led to "
    "the subfield of artificial general intelligence (or AGI), which had several well-funded institutions "
    "by the 2010s. The game of chess has long been viewed as a litmus test for machine intelligence. "
    "Claude Shannon proposed chess-playing as a challenge for AI in 1950, and it became one of the most "
    "studied domains in the history of AI. As with most AI problems, the first chess programs used a "
    "search tree to explore the space of possible games, with the evaluation function and alpha-beta "
    "pruning serving to reduce the number of nodes that need to be evaluated. In 1997, Deep Blue became "
    "the first computer to beat a reigning world chess champion (Gary Kasparov). As the complexity of the "
    "game required an enormous amount of computation, commercial computers at the time were not powerful "
    "enough to play a decent game of chess. The development of specialized hardware and algorithms was "
    "required. In 2011, IBM's Watson computer defeated two former Jeopardy! champions, Brad Rutter and "
    "Ken Jennings, in a demonstration of natural language processing (NLP) and information retrieval. "
    "Watson was a sophisticated system that used over 100 different techniques for analyzing natural "
    "language, identifying sources, finding and generating hypotheses, finding and scoring evidence, and "
    "merging and ranking hypotheses. AlphaGo, an AI system designed by Google DeepMind to play the board "
    "game Go, defeated world champion Lee Sedol four games to one in March 2016. Go had long been "
    "considered to be a grand challenge for AI because the number of possible positions on a Go board "
    "exceeds the number of atoms in the universe. The victory was considered to be a major milestone in "
    "artificial intelligence research. In 2020, OpenAI's GPT-3, a large language model, was released and "
    "demonstrated impressive abilities in generating human-like text. This was followed by the even more "
    "capable GPT-4 in 2023, which showed remarkable performance across a wide range of tasks. The rapid "
    "progress in large language models has sparked both excitement and concern about the future of AI and "
    "its impact on society. Researchers continue to push the boundaries of what's possible, while also "
    "grappling with important questions about safety, alignment, and the ethical implications of "
    "increasingly powerful AI systems. The field continues to evolve rapidly, with new breakthroughs and "
    "applications emerging at an unprecedented pace."
)


@dataclass
class RequestResult:
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # time to first token
    output_tokens: int = 0
    error: str = ""
    inter_token_latencies: list = field(default_factory=list)


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    result = RequestResult()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEXT}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}

    start_time = time.monotonic()
    first_token_time = None
    last_token_time = start_time
    token_count = 0

    async with semaphore:
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:200]}"
                    result.latency = time.monotonic() - start_time
                    return result

                async for line in resp.content:
                    line = line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                now = time.monotonic()
                                if first_token_time is None:
                                    first_token_time = now
                                else:
                                    result.inter_token_latencies.append(now - last_token_time)
                                last_token_time = now
                                token_count += 1
                    except json.JSONDecodeError:
                        pass

            end_time = time.monotonic()
            result.success = True
            result.latency = end_time - start_time
            result.ttft = (first_token_time - start_time) if first_token_time else result.latency
            result.output_tokens = token_count

        except Exception as e:
            result.latency = time.monotonic() - start_time
            result.error = str(e)

    return result


def percentile(data, p):
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


async def run_benchmark(
    url: str,
    model: str,
    num_requests: int,
    rate: float,
    max_tokens: int,
    max_concurrency: int,
):
    print(f"\n{'='*80}")
    print(f"LiteLLM Proxy Streaming Benchmark")
    print(f"{'='*80}")
    print(f"URL:            {url}")
    print(f"Model:          {model}")
    print(f"Requests:       {num_requests}")
    print(f"Rate:           {rate} req/s")
    print(f"Max tokens:     {max_tokens}")
    print(f"Max concurrency:{max_concurrency}")
    print(f"{'='*80}\n")

    semaphore = asyncio.Semaphore(max_concurrency)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=120)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        start_time = time.monotonic()

        for i in range(num_requests):
            task = asyncio.create_task(
                send_streaming_request(session, url, model, max_tokens, semaphore)
            )
            tasks.append(task)

            if (i + 1) % 200 == 0:
                elapsed = time.monotonic() - start_time
                done_count = sum(1 for t in tasks if t.done())
                print(f"  Launched {i+1}/{num_requests} requests ({done_count} completed) [{elapsed:.1f}s]")

            # Rate limiting
            if rate > 0 and i < num_requests - 1:
                await asyncio.sleep(1.0 / rate)

        print(f"\n  All {num_requests} requests launched. Waiting for completion...")
        results = await asyncio.gather(*tasks)

    total_time = time.monotonic() - start_time

    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        print("\nAll requests failed!")
        for r in failed[:5]:
            print(f"  Error: {r.error}")
        return

    latencies = [r.latency for r in successful]
    ttfts = [r.ttft for r in successful]
    output_tokens = [r.output_tokens for r in successful]
    all_itls = []
    for r in successful:
        all_itls.extend(r.inter_token_latencies)

    total_output_tokens = sum(output_tokens)
    rps = len(successful) / total_time

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Time taken:           {total_time:.1f}s")
    print(f"Successful requests:  {len(successful)}/{num_requests}")
    print(f"Failed requests:      {len(failed)}")
    print(f"Request throughput:   {rps:.2f} req/s")
    print(f"Output tok throughput:{total_output_tokens / total_time:.1f} tok/s")
    print(f"")
    print(f"Latency (s):")
    print(f"  Average:   {statistics.mean(latencies):.3f}")
    print(f"  P50:       {percentile(latencies, 50):.3f}")
    print(f"  P95:       {percentile(latencies, 95):.3f}")
    print(f"  P99:       {percentile(latencies, 99):.3f}")
    print(f"  Max:       {max(latencies):.3f}")
    print(f"")
    print(f"TTFT (s):")
    print(f"  Average:   {statistics.mean(ttfts):.3f}")
    print(f"  P50:       {percentile(ttfts, 50):.3f}")
    print(f"  P95:       {percentile(ttfts, 95):.3f}")
    print(f"  P99:       {percentile(ttfts, 99):.3f}")
    print(f"")
    if all_itls:
        print(f"Inter-token latency (s):")
        print(f"  Average:   {statistics.mean(all_itls):.6f}")
        print(f"  P50:       {percentile(all_itls, 50):.6f}")
        print(f"  P99:       {percentile(all_itls, 99):.6f}")
    print(f"")
    print(f"Output tokens per request:")
    print(f"  Average:   {statistics.mean(output_tokens):.1f}")
    print(f"{'='*80}")

    if failed:
        print(f"\nSample errors:")
        for r in failed[:3]:
            print(f"  {r.error}")

    return {
        "total_time": total_time,
        "rps": rps,
        "avg_latency": statistics.mean(latencies),
        "p99_latency": percentile(latencies, 99),
        "avg_ttft": statistics.mean(ttfts),
        "p99_ttft": percentile(ttfts, 99),
        "success_rate": len(successful) / num_requests * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLLM Proxy Benchmark")
    parser.add_argument("--url", default="http://localhost:4000/chat/completions")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=30)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-concurrency", type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            url=args.url,
            model=args.model,
            num_requests=args.num_requests,
            rate=args.rate,
            max_tokens=args.max_tokens,
            max_concurrency=args.max_concurrency,
        )
    )
