"""An owned, source-built proxy whose process env exports AWS_S3_* vars blank.

The shared fixture proxy inherits the harness env, which cannot reproduce a user
shell that exports AWS_S3_ENCRYPTION_KEY_ID / AWS_S3_BUCKET_OWNER as empty
strings. This gateway boots a second proxy with both vars present but blank, so
a batch create through it proves blank means unset, not an empty string.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from e2e_config import unique_marker
from e2e_http import NoBody
from idp import stop_process_group
from proxy_client import ProxyClient, build_proxy_client
from pydantic import TypeAdapter

STARTUP_TIMEOUT_SECONDS: Final = 240
LOG_TAIL_BYTES: Final = 4000


def litellm_root() -> Path:
    spec: Final = importlib.util.find_spec("litellm")
    assert spec is not None and spec.origin is not None, "litellm must be importable to boot the blank-S3-env gateway"
    return Path(spec.origin).resolve().parents[1]


_CONFIG_YAML: Final = """model_list:
  - model_name: bedrock-blank-s3-batch
    litellm_params:
      model: bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION
      s3_region_name: os.environ/AWS_REGION
      s3_bucket_name: os.environ/AWS_BATCH_S3_BUCKET
      s3_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      s3_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_batch_role_arn: os.environ/AWS_BATCH_ROLE_ARN

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
"""


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return TypeAdapter(tuple[str, int]).validate_python(listener.getsockname())[1]


@dataclass(slots=True)
class BedrockEnvGateway:
    base_url: str
    master_key: str
    proxy: ProxyClient
    _environment: Mapping[str, str] = field(repr=False)
    _command: tuple[str, ...] = field(repr=False)
    _log_path: Path
    _child: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    @classmethod
    def start(cls) -> BedrockEnvGateway:
        assert os.environ.get("DATABASE_URL"), "DATABASE_URL is required for the blank-S3-env gateway"
        root: Final = litellm_root()
        port: Final = available_port()
        base_url: Final = f"http://127.0.0.1:{port}"
        master_key: Final = f"sk-e2e-blank-s3-{unique_marker()}"
        directory: Final = Path(tempfile.mkdtemp(prefix="litellm-e2e-blank-s3-"))
        config: Final = directory / "blank-s3-gateway.yaml"
        config.write_text(_CONFIG_YAML)
        environment: Final = {
            **{key: value for key, value in os.environ.items() if not key.startswith("REDIS_")},
            "DATABASE_URL": os.environ["DATABASE_URL"],
            "LITELLM_MASTER_KEY": master_key,
            "STORE_MODEL_IN_DB": "False",
            "PYTHONPATH": str(root),
            "AWS_S3_ENCRYPTION_KEY_ID": "",
            "AWS_S3_BUCKET_OWNER": "",
        }
        gateway: Final = cls(
            base_url=base_url,
            master_key=master_key,
            proxy=build_proxy_client(
                base_url=base_url,
                control_plane_base_url=base_url,
                replica_urls=(base_url,),
                master_key=master_key,
            ),
            _environment=environment,
            _command=(
                sys.executable,
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                str(config),
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
            ),
            _log_path=directory / "blank-s3-gateway.log",
        )
        with gateway._log_path.open("ab") as log:
            gateway._child = subprocess.Popen(
                gateway._command,
                env=dict(gateway._environment),
                stdout=log,
                stderr=log,
                start_new_session=True,
                cwd=root,
            )
        deadline: Final = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            assert gateway._child.poll() is None, f"blank-S3-env gateway exited early; log tail:\n{gateway.log_tail()}"
            result = gateway.proxy.transport.probe("/health/liveliness", params=NoBody())
            if result.status_code == 200:
                return gateway
            time.sleep(0.5)
        tail: Final = gateway.log_tail()
        gateway.stop()
        raise AssertionError(
            f"blank-S3-env gateway did not become ready in {STARTUP_TIMEOUT_SECONDS}s; log tail:\n{tail}"
        )

    def log_tail(self) -> str:
        if not self._log_path.exists():
            return "<no log written>"
        with self._log_path.open("rb") as log:
            log.seek(0, 2)
            size: Final = log.tell()
            log.seek(max(0, size - LOG_TAIL_BYTES))
            return log.read().decode("utf-8", errors="replace")

    def stop(self) -> None:
        if self._child is not None:
            stop_process_group(self._child)
        shutil.rmtree(self._log_path.parent, ignore_errors=True)
