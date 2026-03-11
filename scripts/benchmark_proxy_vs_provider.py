#!/usr/bin/env python3
"""
Benchmark script comparing LiteLLM proxy vs direct provider endpoint.
Makes parallel calls to each endpoint and compares statistics including latency, throughput, and success rates.

USAGE EXAMPLES:

1. Basic Usage (Sequential, Recommended):
   # Set required environment variables
   export LITELLM_PROXY_URL='http://localhost:4000/chat/completions'
   export PROVIDER_URL='https://api.openai.com/v1/chat/completions'
   export LITELLM_PROXY_API_KEY='sk-1234'
   export PROVIDER_API_KEY='sk-openai-key'
   
   # Run from scripts directory
   cd scripts
   python benchmark_proxy_vs_provider.py

2. Multiple Runs for Statistical Accuracy:
   python benchmark_proxy_vs_provider.py --runs 5
   # Averages results across 5 runs for more reliable metrics

3. Realistic Load Testing with Concurrency Limit:
   python benchmark_proxy_vs_provider.py --max-concurrent 100 --requests 2000
   # Limits to 100 concurrent requests (prevents overwhelming the server)

4. Quick Test with Fewer Requests:
   python benchmark_proxy_vs_provider.py --requests 100
   # Faster test with 100 requests instead of default 1000

5. Parallel Execution (Not Recommended):
   python benchmark_proxy_vs_provider.py --parallel
   # Runs both benchmarks simultaneously (may affect accuracy)

6. Custom Timeout:
   python benchmark_proxy_vs_provider.py --timeout 120
   # Sets request timeout to 120 seconds

7. Combined Options:
   python benchmark_proxy_vs_provider.py --runs 3 --requests 500 --max-concurrent 50
   # 3 runs, 500 requests each, max 50 concurrent

REQUIRED ENVIRONMENT VARIABLES:
  - LITELLM_PROXY_URL: Full URL to LiteLLM proxy chat completions endpoint
  - PROVIDER_URL: Full URL to direct provider chat completions endpoint

OPTIONAL ENVIRONMENT VARIABLES:
  - LITELLM_PROXY_API_KEY: API key for LiteLLM proxy (if auth required)
  - PROVIDER_API_KEY: API key for direct provider (if auth required)

OUTPUT:
  The script provides detailed statistics including:
  - Success/error rates
  - Latency metrics (mean, median, p95, p99)
  - Throughput (requests per second)
  - Comparison between proxy and provider performance
  - Run-to-run variance (when using --runs > 1)
"""

import asyncio
import aiohttp
import time
import json
import argparse
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from statistics import mean, median, stdev
import sys
from aiohttp import TCPConnector


@dataclass
class RequestStats:
    """Statistics for a single request"""
    success: bool
    latency: float
    error: str = ""
    status_code: int = 0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    total_time: float = 0.0
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate statistics from the results"""
        if not self.latencies:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": 0.0,
                "error_rate": 1.0,
                "total_time": self.total_time,
                "requests_per_second": 0.0,
                "status_codes": self.status_codes,
                "unique_errors": len(set(self.errors)) if self.errors else 0,
            }
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests) * 100,
            "error_rate": (self.failed_requests / self.total_requests) * 100,
            "total_time": self.total_time,
            "requests_per_second": self.total_requests / self.total_time if self.total_time > 0 else 0,
            "latency_stats": {
                "mean": mean(self.latencies),
                "median": median(self.latencies),
                "min": min(self.latencies),
                "max": max(self.latencies),
                "std_dev": stdev(self.latencies) if len(self.latencies) > 1 else 0.0,
                "p50": median(self.latencies),
                "p95": self._percentile(self.latencies, 95),
                "p99": self._percentile(self.latencies, 99),
            },
            "status_codes": self.status_codes,
            "unique_errors": len(set(self.errors)) if self.errors else 0,
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> RequestStats:
    """Make a single async request and return stats"""
    # Use time.perf_counter() for higher precision timing
    start_time = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
            # Read response body to ensure complete transfer
            response_body = await response.read()
            latency = time.perf_counter() - start_time
            status_code = response.status
            
            if response.status == 200:
                # Validate response is valid JSON
                try:
                    json.loads(response_body)
                except json.JSONDecodeError:
                    return RequestStats(
                        success=False,
                        latency=latency,
                        error="Invalid JSON response",
                        status_code=status_code,
                    )
                
                return RequestStats(
                    success=True,
                    latency=latency,
                    status_code=status_code,
                )
            else:
                error_text = response_body.decode('utf-8', errors='ignore')[:100]
                return RequestStats(
                    success=False,
                    latency=latency,
                    error=f"HTTP {status_code}: {error_text}",
                    status_code=status_code,
                )
    except asyncio.TimeoutError:
        latency = time.perf_counter() - start_time
        return RequestStats(
            success=False,
            latency=latency,
            error="Timeout",
            status_code=0,
        )
    except Exception as e:
        latency = time.perf_counter() - start_time
        return RequestStats(
            success=False,
            latency=latency,
            error=str(e)[:100],
            status_code=0,
        )


async def warmup_endpoint(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    num_warmup: int = 5,
    timeout_seconds: int = 60,
) -> None:
    """Perform warm-up requests to avoid cold start penalties"""
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = TCPConnector(
        limit=100,  # Max connections
        limit_per_host=50,  # Max connections per host
        ttl_dns_cache=300,  # DNS cache TTL
        force_close=False,  # Reuse connections
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            make_request(session, url, headers, payload, timeout)
            for _ in range(num_warmup)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Brief pause after warmup to let connections stabilize
    await asyncio.sleep(0.5)


async def make_request_with_semaphore(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> RequestStats:
    """Make a request with semaphore-based concurrency control"""
    async with semaphore:
        return await make_request(session, url, headers, payload, timeout)


async def benchmark_endpoint(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    num_requests: int = 1000,
    timeout_seconds: int = 60,
    warmup: bool = True,
    max_concurrent: Optional[int] = None,
) -> BenchmarkResults:
    """Benchmark an endpoint with parallel requests
    
    Args:
        url: Endpoint URL to benchmark
        headers: HTTP headers
        payload: Request payload
        num_requests: Total number of requests to make
        timeout_seconds: Request timeout
        warmup: Whether to perform warm-up requests
        max_concurrent: Maximum concurrent requests (None = unlimited, all at once)
    """
    print(f"\nStarting benchmark for {url}")
    
    if warmup:
        print(f"   Warming up with 5 requests...")
        await warmup_endpoint(url, headers, payload, num_warmup=5, timeout_seconds=timeout_seconds)
    
    if max_concurrent:
        print(f"   Making {num_requests} requests with max {max_concurrent} concurrent...")
    else:
        print(f"   Making {num_requests} requests in parallel (unlimited concurrency)...")
    
    results = BenchmarkResults(total_requests=num_requests)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    
    # Set connector limits based on concurrency
    if max_concurrent:
        connector_limit = min(max_concurrent * 2, 200)  # Allow some headroom
        connector_limit_per_host = max_concurrent
    else:
        connector_limit = 200
        connector_limit_per_host = 100
    
    # Use optimized connector for connection pooling and reuse
    connector = TCPConnector(
        limit=connector_limit,
        limit_per_host=connector_limit_per_host,
        ttl_dns_cache=300,  # DNS cache TTL (5 minutes)
        force_close=False,  # Reuse connections for better performance
        enable_cleanup_closed=True,  # Clean up closed connections
    )
    
    # Use time.perf_counter() for higher precision
    start_time = time.perf_counter()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        if max_concurrent:
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [
                make_request_with_semaphore(session, semaphore, url, headers, payload, timeout)
                for _ in range(num_requests)
            ]
        else:
            # Create all tasks at once for maximum parallelism
            tasks = [
                make_request(session, url, headers, payload, timeout)
                for _ in range(num_requests)
            ]
        
        # Execute all requests (with concurrency limit if specified)
        request_stats = await asyncio.gather(*tasks)
    
    results.total_time = time.perf_counter() - start_time
    
    # Aggregate results
    for stats in request_stats:
        if stats.success:
            results.successful_requests += 1
            results.latencies.append(stats.latency)
        else:
            results.failed_requests += 1
            results.errors.append(stats.error)
        
        if stats.status_code > 0:
            results.status_codes[stats.status_code] = results.status_codes.get(stats.status_code, 0) + 1
    
    return results


def print_results(name: str, results: BenchmarkResults):
    """Print formatted benchmark results"""
    stats = results.calculate_stats()
    
    print(f"\n{'='*60}")
    print(f"Results for {name}")
    print(f"{'='*60}")
    print(f"Total Requests:        {stats['total_requests']}")
    print(f"Successful Requests:   {stats['successful_requests']}")
    print(f"Failed Requests:       {stats['failed_requests']}")
    print(f"Success Rate:          {stats['success_rate']:.2f}%")
    print(f"Error Rate:            {stats['error_rate']:.2f}%")
    print(f"Total Time:            {stats['total_time']:.2f}s")
    print(f"Requests/Second:       {stats['requests_per_second']:.2f}")
    
    if 'latency_stats' in stats:
        latency = stats['latency_stats']
        print(f"\nLatency Statistics (seconds):")
        print(f"   Mean:               {latency['mean']:.4f}s")
        print(f"   Median (p50):       {latency['median']:.4f}s")
        print(f"   Min:                {latency['min']:.4f}s")
        print(f"   Max:                {latency['max']:.4f}s")
        print(f"   Std Dev:            {latency['std_dev']:.4f}s")
        print(f"   p95:                {latency['p95']:.4f}s")
        print(f"   p99:                {latency['p99']:.4f}s")
    
    if stats['status_codes']:
        print(f"\nStatus Codes:")
        for code, count in sorted(stats['status_codes'].items()):
            print(f"   {code}: {count}")
    
    if results.errors:
        print(f"\nErrors (showing first 5 unique):")
        unique_errors = list(set(results.errors))[:5]
        for error in unique_errors:
            count = results.errors.count(error)
            print(f"   [{count}x] {error}")


def aggregate_results(results_list: List[BenchmarkResults]) -> BenchmarkResults:
    """Aggregate results from multiple runs"""
    if not results_list:
        return BenchmarkResults()
    
    aggregated = BenchmarkResults()
    
    # Aggregate all latencies
    all_latencies = []
    all_errors = []
    total_requests = 0
    total_successful = 0
    total_failed = 0
    total_time_sum = 0.0
    status_codes_combined = {}
    
    for result in results_list:
        all_latencies.extend(result.latencies)
        all_errors.extend(result.errors)
        total_requests += result.total_requests
        total_successful += result.successful_requests
        total_failed += result.failed_requests
        total_time_sum += result.total_time
        
        for code, count in result.status_codes.items():
            status_codes_combined[code] = status_codes_combined.get(code, 0) + count
    
    aggregated.latencies = all_latencies
    aggregated.errors = all_errors
    aggregated.total_requests = total_requests
    aggregated.successful_requests = total_successful
    aggregated.failed_requests = total_failed
    aggregated.total_time = total_time_sum / len(results_list)  # Average time
    aggregated.status_codes = status_codes_combined
    
    return aggregated


def print_run_variance(name: str, results_list: List[BenchmarkResults]):
    """Print variance statistics across multiple runs"""
    if len(results_list) <= 1:
        return
    
    print(f"\n{'='*60}")
    print(f"Run-to-Run Variance: {name}")
    print(f"{'='*60}")
    
    # Collect mean latencies from each run
    mean_latencies = []
    throughputs = []
    
    for result in results_list:
        stats = result.calculate_stats()
        if 'latency_stats' in stats:
            mean_latencies.append(stats['latency_stats']['mean'])
        throughputs.append(stats['requests_per_second'])
    
    if mean_latencies:
        print(f"\nMean Latency Variance:")
        print(f"   Runs:           {len(mean_latencies)}")
        print(f"   Mean:           {mean(mean_latencies):.4f}s")
        print(f"   Min:            {min(mean_latencies):.4f}s")
        print(f"   Max:            {max(mean_latencies):.4f}s")
        print(f"   Std Dev:        {stdev(mean_latencies):.4f}s" if len(mean_latencies) > 1 else "   Std Dev:        N/A")
        print(f"   Coefficient of Variation: {(stdev(mean_latencies) / mean(mean_latencies) * 100):.2f}%" if len(mean_latencies) > 1 else "   Coefficient of Variation: N/A")
    
    if throughputs:
        print(f"\nThroughput Variance:")
        print(f"   Mean:           {mean(throughputs):.2f} req/s")
        print(f"   Min:            {min(throughputs):.2f} req/s")
        print(f"   Max:            {max(throughputs):.2f} req/s")
        print(f"   Std Dev:        {stdev(throughputs):.2f} req/s" if len(throughputs) > 1 else "   Std Dev:        N/A")


def compare_results(proxy_results: BenchmarkResults, provider_results: BenchmarkResults):
    """Compare and print differences between proxy and provider results"""
    proxy_stats = proxy_results.calculate_stats()
    provider_stats = provider_results.calculate_stats()
    
    print(f"\n{'='*60}")
    print(f"Comparison: LiteLLM Proxy vs Direct Provider")
    print(f"{'='*60}")
    
    # Success Rate Comparison
    print(f"\nSuccess Rate:")
    print(f"   Proxy:   {proxy_stats['success_rate']:.2f}%")
    print(f"   Provider: {provider_stats['success_rate']:.2f}%")
    diff = proxy_stats['success_rate'] - provider_stats['success_rate']
    print(f"   Difference: {diff:+.2f}%")
    
    # Throughput Comparison
    print(f"\nThroughput (requests/second):")
    print(f"   Proxy:   {proxy_stats['requests_per_second']:.2f}")
    print(f"   Provider: {provider_stats['requests_per_second']:.2f}")
    diff = proxy_stats['requests_per_second'] - provider_stats['requests_per_second']
    print(f"   Difference: {diff:+.2f} req/s")
    
    # Latency Comparison
    if 'latency_stats' in proxy_stats and 'latency_stats' in provider_stats:
        print(f"\nLatency Comparison (seconds):")
        proxy_latency = proxy_stats['latency_stats']
        provider_latency = provider_stats['latency_stats']
        
        metrics = ['mean', 'median', 'p95', 'p99']
        for metric in metrics:
            proxy_val = proxy_latency[metric]
            provider_val = provider_latency[metric]
            diff = proxy_val - provider_val
            diff_pct = (diff / provider_val * 100) if provider_val > 0 else 0
            print(f"   {metric.upper():8s}: Proxy={proxy_val:.4f}s, Provider={provider_val:.4f}s, Diff={diff:+.4f}s ({diff_pct:+.2f}%)")
    
    # Total Time Comparison
    print(f"\nTotal Time:")
    print(f"   Proxy:   {proxy_stats['total_time']:.2f}s")
    print(f"   Provider: {provider_stats['total_time']:.2f}s")
    diff = proxy_stats['total_time'] - provider_stats['total_time']
    diff_pct = (diff / provider_stats['total_time'] * 100) if provider_stats['total_time'] > 0 else 0
    print(f"   Difference: {diff:+.2f}s ({diff_pct:+.2f}%)")


async def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(
        description="Benchmark LiteLLM proxy vs direct provider endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (required):
  LITELLM_PROXY_URL    - URL of the LiteLLM proxy endpoint (e.g., http://localhost:4000/chat/completions)
  PROVIDER_URL         - URL of the direct provider endpoint (e.g., https://api.openai.com/v1/chat/completions)
  LITELLM_PROXY_API_KEY - API key for LiteLLM proxy (optional, but may be required)
  PROVIDER_API_KEY     - API key for direct provider (optional, but may be required)

Examples:
  # 1. Basic usage (recommended - sequential execution)
  export LITELLM_PROXY_URL='http://localhost:4000/chat/completions'
  export PROVIDER_URL='https://api.openai.com/v1/chat/completions'
  export LITELLM_PROXY_API_KEY='sk-1234'
  export PROVIDER_API_KEY='sk-openai-key'
  python scripts/benchmark_proxy_vs_provider.py
  
  # 2. Multiple runs for statistical accuracy (recommended)
  python scripts/benchmark_proxy_vs_provider.py --runs 5
  
  # 3. Realistic load testing with concurrency limit
  python scripts/benchmark_proxy_vs_provider.py --max-concurrent 100 --requests 2000
  
  # 4. Quick test with fewer requests
  python scripts/benchmark_proxy_vs_provider.py --requests 100
  
  # 5. Parallel execution (not recommended - may affect accuracy)
  python scripts/benchmark_proxy_vs_provider.py --parallel
  
  # 6. Custom timeout for slower endpoints
  python scripts/benchmark_proxy_vs_provider.py --timeout 120
  
  # 7. Combined options for comprehensive testing
  python scripts/benchmark_proxy_vs_provider.py --runs 3 --requests 500 --max-concurrent 50
  
  # 8. Skip warmup (not recommended - may affect first request accuracy)
  python scripts/benchmark_proxy_vs_provider.py --no-warmup
        """
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run both benchmarks in parallel (default: sequential to avoid interference)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1000,
        help="Number of requests per endpoint (default: 1000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs to average (default: 1, recommended: 3-5 for accuracy)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up requests (not recommended)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent requests (default: unlimited - all at once). "
             "Useful for realistic load testing (e.g., --max-concurrent 100)",
    )
    
    args = parser.parse_args()
    
    # Configuration from environment variables
    LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
    PROVIDER_URL = os.getenv("PROVIDER_URL")
    LITELLM_PROXY_API_KEY = os.getenv("LITELLM_PROXY_API_KEY", "")
    PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY", "")
    
    # Validate required environment variables
    if not LITELLM_PROXY_URL:
        print("Error: LITELLM_PROXY_URL environment variable is required")
        print("   Example: export LITELLM_PROXY_URL='https://your-proxy.com/chat/completions'")
        sys.exit(1)
    
    if not PROVIDER_URL:
        print("Error: PROVIDER_URL environment variable is required")
        print("   Example: export PROVIDER_URL='https://your-provider.com/v1/chat/completions'")
        sys.exit(1)
    
    # Headers for LiteLLM proxy
    proxy_headers = {
        "Content-Type": "application/json",
    }
    if LITELLM_PROXY_API_KEY:
        proxy_headers["Authorization"] = f"Bearer {LITELLM_PROXY_API_KEY}"
    else:
        print("Warning: LITELLM_PROXY_API_KEY not set, requests may fail if authentication is required")
    
    # Headers for direct provider
    provider_headers = {
        "Content-Type": "application/json",
    }
    if PROVIDER_API_KEY:
        provider_headers["Authorization"] = f"Bearer {PROVIDER_API_KEY}"
    else:
        print("Warning: PROVIDER_API_KEY not set, requests may fail if authentication is required")
    
    # Payload (same for both)
    payload = {
        "model": "db-openai-endpoint",  # For proxy
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 100,
        "user": "new_user"
    }
    
    # For direct provider, might need different model name
    provider_payload = payload.copy()
    # provider_payload["model"] = "gpt-3.5-turbo"  # Uncomment if needed
    
    num_requests = args.requests
    timeout_seconds = args.timeout
    
    print("="*60)
    print("LiteLLM Proxy vs Provider Benchmark")
    print("="*60)
    print(f"Configuration (from environment variables):")
    print(f"  Proxy URL:    {LITELLM_PROXY_URL}")
    print(f"  Provider URL: {PROVIDER_URL}")
    print(f"  Proxy API Key: {'Set' if LITELLM_PROXY_API_KEY else 'Not set (may cause auth errors)'}")
    print(f"  Provider API Key: {'Set' if PROVIDER_API_KEY else 'Not set (may cause auth errors)'}")
    print(f"  Requests:     {num_requests}")
    print(f"  Runs:         {args.runs}")
    print(f"  Max Concurrent: {args.max_concurrent if args.max_concurrent else 'Unlimited (all at once)'}")
    print(f"  Timeout:      {timeout_seconds}s")
    print(f"  Warmup:       {'Enabled' if not args.no_warmup else 'Disabled (not recommended)'}")
    print(f"  Mode:         {'Parallel (may affect results)' if args.parallel else 'Sequential (recommended)'}")
    
    if not args.max_concurrent:
        print(f"\nTip: Use --max-concurrent 100 for more realistic load testing")
        print(f"   (prevents overwhelming the server with all requests at once)")
    
    if args.parallel:
        print(f"\nWARNING: Running benchmarks in parallel may affect results due to:")
        print(f"   - Shared network bandwidth")
        print(f"   - Provider endpoint receiving double load (via proxy + direct)")
        print(f"   - Potential rate limiting issues")
        print(f"   - Resource contention")
    
    # Run benchmarks multiple times if requested
    all_proxy_results = []
    all_provider_results = []
    
    warmup_enabled = not args.no_warmup
    
    if args.runs > 1:
        print(f"\nRunning {args.runs} benchmark runs for statistical accuracy...")
        print(f"   Results will be averaged across all runs.\n")
    
    overall_start_time = time.perf_counter()
    
    # Initialize to satisfy type checker (will always be set in loop)
    proxy_results: Optional[BenchmarkResults] = None
    provider_results: Optional[BenchmarkResults] = None
    
    for run_num in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n{'='*60}")
            print(f"Run {run_num}/{args.runs}")
            print(f"{'='*60}")
        
        if args.parallel:
            print(f"\nRunning both benchmarks in parallel...")
            proxy_results, provider_results = await asyncio.gather(
                benchmark_endpoint(
                    LITELLM_PROXY_URL,
                    proxy_headers,
                    payload,
                    num_requests,
                    timeout_seconds,
                    warmup=warmup_enabled and run_num == 1,  # Only warmup on first run
                    max_concurrent=args.max_concurrent,
                ),
                benchmark_endpoint(
                    PROVIDER_URL,
                    provider_headers,
                    provider_payload,
                    num_requests,
                    timeout_seconds,
                    warmup=warmup_enabled and run_num == 1,  # Only warmup on first run
                    max_concurrent=args.max_concurrent,
                ),
            )
        else:
            print(f"\nRunning benchmarks sequentially (proxy first, then provider)...")
            if run_num == 1:
                print(f"   This ensures accurate results without interference.\n")
            
            proxy_results = await benchmark_endpoint(
                LITELLM_PROXY_URL,
                proxy_headers,
                payload,
                num_requests,
                timeout_seconds,
                warmup=warmup_enabled and run_num == 1,  # Only warmup on first run
                max_concurrent=args.max_concurrent,
            )
            
            if run_num < args.runs or args.runs == 1:
                print(f"\nWaiting 3 seconds before starting provider benchmark...")
                await asyncio.sleep(3)  # Longer pause to ensure clean separation
            
            provider_results = await benchmark_endpoint(
                PROVIDER_URL,
                provider_headers,
                provider_payload,
                num_requests,
                timeout_seconds,
                warmup=warmup_enabled and run_num == 1,  # Only warmup on first run
                max_concurrent=args.max_concurrent,
            )
        
        all_proxy_results.append(proxy_results)
        all_provider_results.append(provider_results)
        
        # Brief pause between runs
        if run_num < args.runs:
            print(f"\nWaiting 5 seconds before next run...")
            await asyncio.sleep(5)
    
    overall_benchmark_time = time.perf_counter() - overall_start_time
    print(f"\nAll benchmark runs completed in {overall_benchmark_time:.2f}s")
    
    # Aggregate results across multiple runs
    if args.runs > 1:
        final_proxy_results = aggregate_results(all_proxy_results)
        final_provider_results = aggregate_results(all_provider_results)
        print(f"\nAggregated results across {args.runs} runs:")
    else:
        # Use results from single run
        if proxy_results is None or provider_results is None:
            raise RuntimeError("Benchmark results not initialized")
        final_proxy_results = proxy_results
        final_provider_results = provider_results
        print(f"\nResults:")
    
    # Print individual results
    print_results("LiteLLM Proxy", final_proxy_results)
    print_results("Direct Provider", final_provider_results)
    
    # Print comparison
    compare_results(final_proxy_results, final_provider_results)
    
    # Show run-to-run variance if multiple runs
    if args.runs > 1:
        print_run_variance("LiteLLM Proxy", all_proxy_results)
        print_run_variance("Direct Provider", all_provider_results)
    
    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

