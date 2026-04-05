import requests
import argparse
import time
import concurrent.futures
import threading


LITELLM_BASE_URL = "***/v1/files/test-file-65/content"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-1234",
}

def percentile(values, p):
    """Returns percentile value using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def render_progress(completed, total):
    """Renders a simple terminal progress bar."""
    bar_width = 32
    fraction = completed / total if total else 1
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = fraction * 100
    return f"[{bar}] {completed}/{total} ({percent:6.2f}%)"


def make_request(session, url):
    """Makes a single request and returns timing + success info."""

    start_time = time.time()
    try:
        response = session.get(
            url,
            headers=HEADERS,
        )
        response.raise_for_status()
        success = True
    except requests.exceptions.RequestException as e:
        success = False
        error_message = str(e)

    response_time = time.time() - start_time
    if success:
        return success, response_time, ""

    return success, response_time, error_message

def run_load_test(num_requests):
    """Runs the load test by making a specified number of requests in parallel."""
    print(f"Starting load test with {num_requests} parallel requests to {LITELLM_BASE_URL}")

    lock = threading.Lock()
    completed = 0
    errors = 0
    latencies = []
    first_error = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with requests.Session() as session:
            futures = [executor.submit(make_request, session, LITELLM_BASE_URL) for _ in range(num_requests)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    success, response_time, error_message = future.result()
                    with lock:
                        completed += 1
                        latencies.append(response_time)
                        if not success:
                            errors += 1
                            if first_error is None:
                                first_error = error_message

                        progress = render_progress(completed, num_requests)
                        print(f"\r{progress}", end="", flush=True)
                except Exception as exc:
                    with lock:
                        completed += 1
                        errors += 1
                        if first_error is None:
                            first_error = str(exc)
                        progress = render_progress(completed, num_requests)
                        print(f"\r{progress}", end="", flush=True)

    print()  # move to the next line after progress bar

    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    print("Load test finished.")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {num_requests - errors}")
    print(f"Errors: {errors}")
    print(f"Latency p50: {p50:.3f}s")
    print(f"Latency p90: {p90:.3f}s")
    print(f"Latency p95: {p95:.3f}s")
    print(f"Latency p99: {p99:.3f}s")

    if first_error:
        print(f"First error: {first_error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script to run a load test against a URL.")
    parser.add_argument("num_requests", type=int, help="The number of requests to make.")
    
    args = parser.parse_args()
    
    run_load_test(args.num_requests)