#!/usr/bin/env python3
"""
Validate proxy overhead for RPS, TTFT/latency, and token throughput.

Examples:
  BASELINE_URL=http://localhost:4000/v1/chat/completions \
  CANDIDATE_URL=http://localhost:4001/v1/chat/completions \
  BASELINE_API_KEY=sk-1234 CANDIDATE_API_KEY=sk-1234 \
  python scripts/validate_proxy_overhead.py --stream --requests 500 --concurrency 100

  PROVIDER_URL=https://api.openai.com/v1/chat/completions \
  LITELLM_PROXY_URL=http://localhost:4000/v1/chat/completions \
  PROVIDER_API_KEY=sk-... LITELLM_PROXY_API_KEY=sk-1234 \
  python scripts/validate_proxy_overhead.py --payload-file payload.json
"""

import argparse
import concurrent.futures
import json
import os
import ssl
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_PAYLOAD = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Reply with a single short sentence."}],
    "max_tokens": 32,
}


@dataclass
class RequestResult:
    ok: bool
    total_seconds: float
    ttft_seconds: Optional[float] = None
    total_tokens: int = 0
    error: str = ""


@dataclass
class RunResult:
    label: str
    wall_seconds: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successes(self) -> List[RequestResult]:
        return [result for result in self.results if result.ok]

    def stats(self) -> Dict[str, Any]:
        successes = self.successes
        latencies = [result.total_seconds * 1000 for result in successes]
        ttfts = [
            result.ttft_seconds * 1000
            for result in successes
            if result.ttft_seconds is not None
        ]
        total_tokens = sum(result.total_tokens for result in successes)
        return {
            "requests": len(self.results),
            "successes": len(successes),
            "failures": len(self.results) - len(successes),
            "rps": len(successes) / self.wall_seconds if self.wall_seconds else 0.0,
            "tpm": (
                (total_tokens / self.wall_seconds * 60) if self.wall_seconds else 0.0
            ),
            "latency_ms": percentile_stats(latencies),
            "ttft_ms": percentile_stats(ttfts),
        }


def percentile_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = sorted(values)
    return {
        "p50": values[int((len(values) - 1) * 0.50)],
        "p95": values[int((len(values) - 1) * 0.95)],
        "p99": values[int((len(values) - 1) * 0.99)],
    }


def percent_delta(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (candidate - baseline) / baseline * 100


def auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: float,
    stream: bool,
) -> RequestResult:
    return (
        post_stream_request(url, headers, payload, timeout_seconds)
        if stream
        else post_json_request(url, headers, payload, timeout_seconds)
    )


def make_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> urllib.request.Request:
    body = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url=url, data=body, headers=headers, method="POST")


def post_json_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> RequestResult:
    start = time.perf_counter()
    try:
        request = make_request(url, headers, payload, timeout_seconds)
        with urllib.request.urlopen(
            request, timeout=timeout_seconds, context=ssl.create_default_context()
        ) as response:
            body = response.read()
        total_seconds = time.perf_counter() - start
        data = json.loads(body)
        usage = data.get("usage") or {}
        return RequestResult(
            ok=True,
            total_seconds=total_seconds,
            total_tokens=int(usage.get("total_tokens") or 0),
        )
    except urllib.error.HTTPError as exc:
        body = exc.read()[:200]
        return RequestResult(
            ok=False,
            total_seconds=time.perf_counter() - start,
            error=f"HTTP {exc.code}: {body!r}",
        )
    except Exception as exc:
        return RequestResult(
            ok=False,
            total_seconds=time.perf_counter() - start,
            error=str(exc)[:200],
        )


def post_stream_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> RequestResult:
    stream_payload = {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft_seconds: Optional[float] = None
    total_tokens = 0
    try:
        request = make_request(url, headers, stream_payload, timeout_seconds)
        with urllib.request.urlopen(
            request, timeout=timeout_seconds, context=ssl.create_default_context()
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft_seconds is None:
                    ttft_seconds = time.perf_counter() - start
                usage = chunk.get("usage") or {}
                if usage.get("total_tokens") is not None:
                    total_tokens = int(usage["total_tokens"])

        return RequestResult(
            ok=True,
            total_seconds=time.perf_counter() - start,
            ttft_seconds=ttft_seconds,
            total_tokens=total_tokens,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read()[:200]
        return RequestResult(
            ok=False,
            total_seconds=time.perf_counter() - start,
            ttft_seconds=ttft_seconds,
            error=f"HTTP {exc.code}: {body!r}",
        )
    except Exception as exc:
        return RequestResult(
            ok=False,
            total_seconds=time.perf_counter() - start,
            ttft_seconds=ttft_seconds,
            error=str(exc)[:200],
        )


def run_endpoint(
    label: str,
    url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    requests: int,
    concurrency: int,
    stream: bool,
    timeout_seconds: float,
) -> RunResult:
    headers = auth_headers(api_key)
    for _ in range(min(10, requests)):
        post_request(url, headers, payload, timeout_seconds, stream)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_request,
                url,
                headers,
                payload,
                timeout_seconds,
                stream,
            )
            for _ in range(requests)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    wall_seconds = time.perf_counter() - start
    return RunResult(label=label, wall_seconds=wall_seconds, results=results)


def print_stats(baseline: RunResult, candidate: RunResult) -> None:
    baseline_stats = baseline.stats()
    candidate_stats = candidate.stats()

    print("\nMetric                        Baseline        Candidate       Delta")
    print("-" * 68)
    for metric in ("rps", "tpm"):
        base = baseline_stats[metric]
        cand = candidate_stats[metric]
        print(
            f"{metric.upper():<28}{base:>10.2f}      {cand:>10.2f}      {percent_delta(cand, base):>7.2f}%"
        )

    for family in ("latency_ms", "ttft_ms"):
        if family == "ttft_ms" and candidate_stats[family]["p50"] == 0:
            continue
        for percentile in ("p50", "p95", "p99"):
            base = baseline_stats[family][percentile]
            cand = candidate_stats[family][percentile]
            print(
                f"{family}.{percentile:<19}{base:>10.2f} ms   {cand:>10.2f} ms   {percent_delta(cand, base):>7.2f}%"
            )

    print("-" * 68)
    print(
        f"{baseline.label}: {baseline_stats['successes']}/{baseline_stats['requests']} successes, "
        f"{baseline_stats['failures']} failures"
    )
    print(
        f"{candidate.label}: {candidate_stats['successes']}/{candidate_stats['requests']} successes, "
        f"{candidate_stats['failures']} failures"
    )


def load_payload(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return dict(DEFAULT_PAYLOAD)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-url", default=os.getenv("BASELINE_URL") or os.getenv("PROVIDER_URL")
    )
    parser.add_argument(
        "--candidate-url",
        default=os.getenv("CANDIDATE_URL") or os.getenv("LITELLM_PROXY_URL"),
    )
    parser.add_argument(
        "--baseline-api-key",
        default=os.getenv("BASELINE_API_KEY") or os.getenv("PROVIDER_API_KEY"),
    )
    parser.add_argument(
        "--candidate-api-key",
        default=os.getenv("CANDIDATE_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY"),
    )
    parser.add_argument("--payload-file")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--stream", action="store_true", help="Measure TTFT using streaming responses"
    )
    args = parser.parse_args()

    if not args.baseline_url or not args.candidate_url:
        raise SystemExit(
            "Set --baseline-url and --candidate-url, or BASELINE_URL/CANDIDATE_URL."
        )

    payload = load_payload(args.payload_file)
    baseline = run_endpoint(
        label="baseline",
        url=args.baseline_url,
        api_key=args.baseline_api_key,
        payload=payload,
        requests=args.requests,
        concurrency=args.concurrency,
        stream=args.stream,
        timeout_seconds=args.timeout,
    )
    candidate = run_endpoint(
        label="candidate",
        url=args.candidate_url,
        api_key=args.candidate_api_key,
        payload=payload,
        requests=args.requests,
        concurrency=args.concurrency,
        stream=args.stream,
        timeout_seconds=args.timeout,
    )
    print_stats(baseline, candidate)


if __name__ == "__main__":
    main()
