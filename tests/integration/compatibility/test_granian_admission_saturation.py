import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests._master_key import MASTER_KEY
from tests.integration._support.process import stop_child
from tests.integration._support.ports import allocate_port

MAX_IN_FLIGHT: Final = 8
MAX_QUEUED: Final = 8
REQUESTS: Final = 200


@pytest.mark.asyncio
async def test_granian_admission_control_saturation(tmp_path: Path) -> None:
    fake_port: Final = allocate_port()
    proxy_port: Final = allocate_port()
    fake_script: Final = Path(__file__).parents[2] / "_fake_openai_endpoint_server.py"
    config_path: Final = tmp_path / "saturation_config.yaml"
    proxy_env: Final = {**{k: v for k, v in os.environ.items() if k != "DATABASE_URL"}, "LITELLM_MASTER_KEY": MASTER_KEY}
    config_path.write_text(
        f"""model_list:
  - model_name: slow-endpoint
    litellm_params:
      model: openai/slow-endpoint
      api_base: http://127.0.0.1:{fake_port}/v1
      api_key: fake-upstream-key
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  max_in_flight_requests_per_worker: {MAX_IN_FLIGHT}
  max_queued_requests_per_worker: {MAX_QUEUED}
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
            env=proxy_env,
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

                async def send_completion() -> tuple[int, bool]:
                    response = await client.post(
                        "/chat/completions",
                        headers={"Authorization": f"Bearer {MASTER_KEY}"},
                        json={
                            "model": "slow-endpoint",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                        timeout=10,
                    )
                    return response.status_code, "retry-after" in response.headers

                results: Final = await asyncio.gather(*(send_completion() for _ in range(REQUESTS)))

            statuses: Final = [result[0] for result in results]
            rejected: Final = [result for result in results if result[0] == 503]
            assert set(statuses) <= {200, 503}, statuses
            assert 1 <= statuses.count(200) <= MAX_IN_FLIGHT + MAX_QUEUED, statuses
            assert len(rejected) >= REQUESTS - MAX_IN_FLIGHT - MAX_QUEUED, statuses
            assert all(result[1] for result in rejected), rejected
        finally:
            stop_child(proxy_process)
    finally:
        stop_child(fake_process)
