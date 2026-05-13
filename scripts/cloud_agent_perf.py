#!/usr/bin/env python3
"""
Minimal LiteLLM proxy performance runner for Cloud-agent verification.

Measures proxy throughput (RPS) and, when a direct endpoint is provided,
the proxy latency overhead compared with that direct endpoint.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp


DEFAULT_PROXY_URL = "http://localhost:4000/chat/completions"
DEFAULT_MODEL = "fake-openai-endpoint"
DEFAULT_MESSAGES = [{"role": "user", "content": "Say hello in one sentence."}]


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    url: str
    model: str
    api_key: Optional[str] = None


@dataclass(frozen=True)
class RequestSample:
    success: bool
    latency_s: float
    status_code: int = 0
    error: str = ""


@dataclass
class LoadTestResult:
    endpoint_name: str
    total_requests: int
    wall_time_s: float
    samples: List[RequestSample] = field(default_factory=list)

    @property
    def successful_requests(self) -> int:
        return sum(1 for sample in self.samples if sample.success)

    @property
    def failed_requests(self) -> int:
        return self.total_requests - self.successful_requests

    @property
    def successful_latencies_s(self) -> List[float]:
        return [sample.latency_s for sample in self.samples if sample.success]

    def to_summary(self) -> Dict[str, Any]:
        latencies = self.successful_latencies_s
        summary: Dict[str, Any] = {
            "endpoint": self.endpoint_name,
            "requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "wall_time_s": self.wall_time_s,
            "rps": (
                self.total_requests / self.wall_time_s if self.wall_time_s > 0 else 0.0
            ),
            "successful_rps": (
                self.successful_requests / self.wall_time_s
                if self.wall_time_s > 0
                else 0.0
            ),
        }
        if latencies:
            summary["latency_ms"] = {
                "mean": statistics.mean(latencies) * 1000,
                "p50": percentile(latencies, 50) * 1000,
                "p95": percentile(latencies, 95) * 1000,
                "p99": percentile(latencies, 99) * 1000,
                "min": min(latencies) * 1000,
                "max": max(latencies) * 1000,
            }
        return summary


def percentile(values: List[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = round((percentile_value / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def headers_for(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def parse_json_arg(raw_value: Optional[str], label: str) -> Optional[Any]:
    if raw_value is None:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc


def build_payload(
    model: str,
    messages_json: Optional[str] = None,
    payload_json: Optional[str] = None,
) -> Dict[str, Any]:
    custom_payload = parse_json_arg(payload_json, "--payload-json")
    if custom_payload is not None:
        if not isinstance(custom_payload, dict):
            raise ValueError("--payload-json must decode to a JSON object")
        return {**custom_payload, "model": custom_payload.get("model", model)}

    messages = parse_json_arg(messages_json, "--messages-json") or DEFAULT_MESSAGES
    if not isinstance(messages, list):
        raise ValueError("--messages-json must decode to a JSON array")
    return {"model": model, "messages": messages, "max_tokens": 16}


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: EndpointConfig,
    payload: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
    validate_json: bool,
) -> RequestSample:
    start = time.perf_counter()
    try:
        async with session.post(
            endpoint.url,
            headers=headers_for(endpoint.api_key),
            json=payload,
            timeout=timeout,
        ) as response:
            body = await response.read()
            latency_s = time.perf_counter() - start
            if not 200 <= response.status < 300:
                body_preview = body.decode("utf-8", errors="replace")[:200]
                return RequestSample(
                    success=False,
                    latency_s=latency_s,
                    status_code=response.status,
                    error=f"HTTP {response.status}: {body_preview}",
                )
            if validate_json:
                try:
                    json.loads(body)
                except json.JSONDecodeError as exc:
                    return RequestSample(
                        success=False,
                        latency_s=latency_s,
                        status_code=response.status,
                        error=f"invalid JSON response: {exc}",
                    )
            return RequestSample(
                success=True,
                latency_s=latency_s,
                status_code=response.status,
            )
    except Exception as exc:
        return RequestSample(
            success=False,
            latency_s=time.perf_counter() - start,
            error=str(exc)[:200],
        )


async def run_batch(
    endpoint: EndpointConfig,
    payload: Dict[str, Any],
    request_count: int,
    concurrency: int,
    timeout_s: float,
    validate_json: bool,
) -> List[RequestSample]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(
        limit=max(concurrency * 2, 1),
        limit_per_host=max(concurrency, 1),
        ttl_dns_cache=300,
        force_close=False,
        enable_cleanup_closed=True,
    )

    async def guarded_request(session: aiohttp.ClientSession) -> RequestSample:
        async with semaphore:
            return await send_request(
                session=session,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
                validate_json=validate_json,
            )

    async with aiohttp.ClientSession(connector=connector) as session:
        return await asyncio.gather(
            *[guarded_request(session) for _ in range(request_count)]
        )


async def run_load_test(
    endpoint: EndpointConfig,
    payload: Dict[str, Any],
    requests: int,
    concurrency: int,
    warmup_requests: int,
    timeout_s: float,
    validate_json: bool = True,
) -> LoadTestResult:
    if requests < 1:
        raise ValueError("--requests must be at least 1")
    if concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    if warmup_requests > 0:
        await run_batch(
            endpoint=endpoint,
            payload=payload,
            request_count=warmup_requests,
            concurrency=min(concurrency, warmup_requests),
            timeout_s=timeout_s,
            validate_json=validate_json,
        )

    start = time.perf_counter()
    samples = await run_batch(
        endpoint=endpoint,
        payload=payload,
        request_count=requests,
        concurrency=concurrency,
        timeout_s=timeout_s,
        validate_json=validate_json,
    )
    wall_time_s = time.perf_counter() - start
    return LoadTestResult(
        endpoint_name=endpoint.name,
        total_requests=requests,
        wall_time_s=wall_time_s,
        samples=list(samples),
    )


def compare_overhead(
    proxy_summary: Dict[str, Any], direct_summary: Dict[str, Any]
) -> Dict[str, float]:
    proxy_latency = proxy_summary.get("latency_ms", {})
    direct_latency = direct_summary.get("latency_ms", {})
    comparison: Dict[str, float] = {}
    for metric in ("mean", "p50", "p95", "p99"):
        proxy_value = proxy_latency.get(metric)
        direct_value = direct_latency.get(metric)
        if proxy_value is None or direct_value is None:
            continue
        overhead_ms = proxy_value - direct_value
        comparison[f"{metric}_overhead_ms"] = overhead_ms
        comparison[f"{metric}_overhead_pct"] = (
            (overhead_ms / direct_value) * 100 if direct_value > 0 else 0.0
        )
    return comparison


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\n{summary['endpoint']}")
    print(f"  requests:       {summary['requests']}")
    print(f"  successes:      {summary['successful_requests']}")
    print(f"  failures:       {summary['failed_requests']}")
    print(f"  wall time:      {summary['wall_time_s']:.2f}s")
    print(f"  rps:            {summary['rps']:.2f}")
    print(f"  successful rps: {summary['successful_rps']:.2f}")
    latency = summary.get("latency_ms")
    if latency:
        print(f"  mean latency:   {latency['mean']:.2f} ms")
        print(f"  p50 latency:    {latency['p50']:.2f} ms")
        print(f"  p95 latency:    {latency['p95']:.2f} ms")
        print(f"  p99 latency:    {latency['p99']:.2f} ms")


def print_overhead(overhead: Dict[str, float]) -> None:
    if not overhead:
        return
    print("\nProxy overhead vs direct endpoint")
    for metric in ("mean", "p50", "p95", "p99"):
        ms_key = f"{metric}_overhead_ms"
        pct_key = f"{metric}_overhead_pct"
        if ms_key in overhead:
            print(f"  {metric}: {overhead[ms_key]:+.2f} ms ({overhead[pct_key]:+.2f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure LiteLLM proxy RPS and optional proxy overhead.",
    )
    parser.add_argument(
        "--proxy-url",
        default=os.getenv("LITELLM_PROXY_URL", DEFAULT_PROXY_URL),
        help="LiteLLM /chat/completions URL.",
    )
    parser.add_argument(
        "--proxy-api-key",
        default=os.getenv("LITELLM_PROXY_API_KEY", "sk-1234"),
        help="LiteLLM proxy API key.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LITELLM_PERF_MODEL", DEFAULT_MODEL),
        help="Model name to send to the proxy.",
    )
    parser.add_argument(
        "--direct-url",
        default=os.getenv("LITELLM_DIRECT_URL"),
        help="Optional direct provider /chat/completions URL for overhead comparison.",
    )
    parser.add_argument(
        "--direct-api-key",
        default=os.getenv("LITELLM_DIRECT_API_KEY"),
        help="Optional direct provider API key.",
    )
    parser.add_argument(
        "--direct-model",
        default=os.getenv("LITELLM_DIRECT_MODEL"),
        help="Optional model name to send to the direct provider.",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--messages-json",
        help="JSON array of OpenAI-compatible chat messages.",
    )
    parser.add_argument(
        "--payload-json",
        help="Full JSON request body. The model field is filled from --model if absent.",
    )
    parser.add_argument(
        "--no-validate-json",
        action="store_true",
        help="Do not require 2xx responses to contain JSON.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write machine-readable results.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    validate_json = not args.no_validate_json
    proxy_endpoint = EndpointConfig(
        name="LiteLLM proxy",
        url=args.proxy_url,
        model=args.model,
        api_key=args.proxy_api_key,
    )
    proxy_payload = build_payload(
        model=args.model,
        messages_json=args.messages_json,
        payload_json=args.payload_json,
    )

    proxy_result = await run_load_test(
        endpoint=proxy_endpoint,
        payload=proxy_payload,
        requests=args.requests,
        concurrency=args.concurrency,
        warmup_requests=args.warmup_requests,
        timeout_s=args.timeout,
        validate_json=validate_json,
    )
    proxy_summary = proxy_result.to_summary()
    print_summary(proxy_summary)

    result_doc: Dict[str, Any] = {"proxy": proxy_summary}

    if args.direct_url:
        direct_model = args.direct_model or args.model
        direct_endpoint = EndpointConfig(
            name="Direct endpoint",
            url=args.direct_url,
            model=direct_model,
            api_key=args.direct_api_key,
        )
        direct_payload = {**proxy_payload, "model": direct_model}
        direct_result = await run_load_test(
            endpoint=direct_endpoint,
            payload=direct_payload,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup_requests=args.warmup_requests,
            timeout_s=args.timeout,
            validate_json=validate_json,
        )
        direct_summary = direct_result.to_summary()
        overhead = compare_overhead(proxy_summary, direct_summary)
        print_summary(direct_summary)
        print_overhead(overhead)
        result_doc["direct"] = direct_summary
        result_doc["overhead"] = overhead

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(result_doc, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"\nWrote JSON results to {args.output_json}")

    return 0 if proxy_result.failed_requests == 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
