from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import pytest
from e2e_config import MASTER_KEY
from e2e_http import NoBody, unwrap
from e2e_metadata import Domain, Subject, meta
from models import CostMap
from pydantic import BaseModel, ConfigDict
from transport import HttpTransport


class CostMapSourceResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    source: str
    model_count: int


class TestModelCostPreloadRevalidation:
    @pytest.mark.parametrize("launch_as_module", [False, True], ids=("console", "module"))
    @pytest.mark.e2e
    @pytest.mark.owned_gateway
    @pytest.mark.quiet_stack
    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_preloaded_gunicorn_workers_revalidate_without_master_fetch(
        self,
        tmp_path: Path,
        launch_as_module: bool,
    ) -> None:
        assert "DATABASE_URL" in os.environ
        request_paths: Final[queue.Queue[str]] = queue.Queue()
        catalog_path: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
        catalog: Final = json.loads(catalog_path.read_text())
        catalog["preload/e2e-model"] = {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
        }
        response_body: Final = json.dumps(catalog).encode()

        class CatalogHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                request_paths.put(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        catalog_server: Final = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
        catalog_thread: Final = threading.Thread(target=catalog_server.serve_forever, daemon=True)
        catalog_thread.start()
        catalog_url: Final = f"http://127.0.0.1:{catalog_server.server_port}/prices.json"

        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            proxy_port: Final = port_socket.getsockname()[1]

        marker_dir: Final = tmp_path / "workers"
        marker_dir.mkdir()
        master_pid_path: Final = tmp_path / "master.pid"
        gunicorn_config: Final = tmp_path / "gunicorn.py"
        gunicorn_config.write_text(
            "\n".join(
                (
                    "import json",
                    "import os",
                    "import time",
                    "from pathlib import Path",
                    "",
                    "def on_starting(server):",
                    f"    Path({str(master_pid_path)!r}).write_text(str(server.pid))",
                    "",
                    "def post_worker_init(worker):",
                    "    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info",
                    "    from litellm.litellm_core_utils.model_cost_revalidate import REVALIDATOR",
                    "    revalidator = REVALIDATOR._thread",
                    "    if revalidator is not None:",
                    "        revalidator.join(timeout=60)",
                    "    source = get_model_cost_map_source_info()['source']",
                    f"    marker = Path({str(marker_dir)!r}) / f'{{worker.pid}}.json'",
                    "    marker.write_text(json.dumps({'pid': worker.pid, 'source': source}))",
                )
            )
        )

        log_file: Final = (tmp_path / "gunicorn.log").open("w")
        gunicorn_executable: Final = Path(sys.executable).with_name("gunicorn")
        gunicorn_args: Final = (
            "litellm.proxy.proxy_server:app",
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            "--preload",
            "--workers",
            "2",
            "--bind",
            f"127.0.0.1:{proxy_port}",
            "--config",
            str(gunicorn_config),
        )
        command: Final = (
            (sys.executable, "-m", "gunicorn", *gunicorn_args)
            if launch_as_module
            else (str(gunicorn_executable), *gunicorn_args)
        )
        process: Final = subprocess.Popen(
            command,
            env={
                **os.environ,
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "LITELLM_LOCAL_MODEL_COST_MAP": "False",
                "LITELLM_MODEL_COST_MAP_URL": catalog_url,
            },
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline: Final = time.monotonic() + 120
            worker_markers: tuple[Path, ...] = ()
            while time.monotonic() < deadline and len(worker_markers) < 2:
                assert process.poll() is None, (tmp_path / "gunicorn.log").read_text()
                worker_markers = tuple(marker_dir.glob("*.json"))
                if len(worker_markers) < 2:
                    time.sleep(0.1)

            assert master_pid_path.exists(), (tmp_path / "gunicorn.log").read_text()
            assert len(worker_markers) == 2, (tmp_path / "gunicorn.log").read_text()
            master_pid: Final = int(master_pid_path.read_text())
            worker_states: Final = tuple(json.loads(path.read_text()) for path in worker_markers)
            assert {state["source"] for state in worker_states} == {"remote"}
            assert len({state["pid"] for state in worker_states}) == 2
            assert master_pid not in {state["pid"] for state in worker_states}
            assert request_paths.qsize() == 2

            transport: Final = HttpTransport(
                base_url=f"http://127.0.0.1:{proxy_port}",
                master_key=MASTER_KEY,
            )
            source: Final = unwrap(
                transport.get(
                    "/model/cost_map/source",
                    headers=transport.master,
                    params=NoBody(),
                    response_type=CostMapSourceResponse,
                )
            )
            price_map: Final = unwrap(
                transport.get(
                    "/public/litellm_model_cost_map",
                    headers=transport.master,
                    params=NoBody(),
                    response_type=CostMap,
                )
            )
            assert source.source == "remote"
            assert source.model_count > 0
            assert price_map.root["preload/e2e-model"].input_cost_per_token == 0.000001
            assert request_paths.qsize() == 2
        finally:
            catalog_server.shutdown()
            catalog_server.server_close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            catalog_thread.join(timeout=10)
            log_file.close()
