import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LITELLM_RUN_SATURATION_BENCHMARK") != "1",
    reason="set LITELLM_RUN_SATURATION_BENCHMARK=1 to run the saturation benchmark",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _percentile(values: list[float], percentile: float) -> float:
    return sorted(values)[min(int(len(values) * percentile), len(values) - 1)]


@pytest.mark.asyncio
async def test_granian_admission_control_saturation(tmp_path: Path) -> None:
    fake_port: Final = _free_port()
    proxy_port: Final = _free_port()
    fake_script: Final = Path(__file__).parents[1] / "_fake_openai_endpoint_server.py"
    config_path: Final = tmp_path / "saturation_config.yaml"
    config_path.write_text(
        f"""model_list:
  - model_name: slow-endpoint
    litellm_params:
      model: openai/slow-endpoint
      api_base: http://127.0.0.1:{fake_port}/v1
general_settings:
  master_key: sk-saturation
  max_in_flight_requests_per_worker: 8
  max_queued_requests_per_worker: 8
  admission_queue_timeout_seconds: 0.5
"""
    )
    fake_process: Final = subprocess.Popen(
        [sys.executable, str(fake_script), "--host", "127.0.0.1", "--port", str(fake_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proxy_process: Final = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                str(config_path),
                "--run_granian",
                "--num_workers",
                "1",
                "--port",
                str(proxy_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{proxy_port}") as client:
                deadline: Final = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        response: Final = await client.get("/health/liveliness", timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.25)
                else:
                    raise AssertionError("Granian proxy did not become healthy")

                liveness_latencies: Final[list[float]] = []
                stop_sampling: Final = asyncio.Event()

                async def sample_liveness() -> None:
                    while not stop_sampling.is_set():
                        start: Final = time.perf_counter()
                        try:
                            response = await client.get("/health/liveliness", timeout=2)
                            response.raise_for_status()
                            liveness_latencies.append(time.perf_counter() - start)
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.05)

                async def send_completion() -> tuple[int, float, bool]:
                    start: Final = time.perf_counter()
                    response = await client.post(
                        "/chat/completions",
                        headers={"Authorization": "Bearer sk-saturation"},
                        json={
                            "model": "slow-endpoint",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                        timeout=10,
                    )
                    return response.status_code, time.perf_counter() - start, "retry-after" in response.headers

                sampler: Final = asyncio.create_task(sample_liveness())
                results: Final = await asyncio.gather(*(send_completion() for _ in range(200)))
                stop_sampling.set()
                await sampler

            statuses: Final = [result[0] for result in results]
            latencies: Final = [result[1] for result in results]
            rejected: Final = [result for result in results if result[0] == 503]
            assert set(statuses) <= {200, 503}
            assert rejected
            assert all(result[2] for result in rejected)
            assert _percentile(latencies, 0.99) < 5
            assert liveness_latencies
            assert _percentile(liveness_latencies, 0.95) < 0.5

            duration: Final = max(latencies)
            print(
                "\nmetric          value\n"
                f"rps             {len(results) / duration:.2f}\n"
                f"200 count       {statuses.count(200)}\n"
                f"503 count       {statuses.count(503)}\n"
                f"p50             {_percentile(latencies, 0.50):.3f}s\n"
                f"p95             {_percentile(latencies, 0.95):.3f}s\n"
                f"p99             {_percentile(latencies, 0.99):.3f}s\n"
                f"liveness p95    {_percentile(liveness_latencies, 0.95):.3f}s"
            )
        finally:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proxy_process.kill()
                proxy_process.wait()
    finally:
        fake_process.terminate()
        try:
            fake_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fake_process.kill()
            fake_process.wait()
