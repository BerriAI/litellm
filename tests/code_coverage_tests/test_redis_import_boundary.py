import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_REDIS_IMPORTS: Final = (
    "import redis\n\nredis.Redis()\n",
    "import redis as client\n\nclient.Redis()\n",
    "from redis import Redis\n\nRedis()\n",
    "from redis import Redis as Client\n\nClient()\n",
    "import redis.asyncio\n\nredis.asyncio.Redis()\n",
    "import redis.asyncio as client\n\nclient.Redis()\n",
    "from redis import asyncio\n\nasyncio.Redis()\n",
    "from redis.asyncio import Redis as Client\n\nClient()\n",
)
_CALLERS: Final = (
    "litellm/proxy/_redis_boundary_probe.py",
    "litellm/redis/stores/_redis_boundary_probe.py",
    "litellm/redis/transport/_redis_boundary_probe.py",
    "litellm/caching/_redis_boundary_probe.py",
    "litellm/caching/nested/redis_cache.py",
    "enterprise/litellm_enterprise/_redis_boundary_probe.py",
)
_EXCEPTIONS: Final = (
    "litellm/redis/transport/redis_py.py",
    "litellm/caching/redis_cache.py",
    "litellm/caching/redis_cluster_cache.py",
    "litellm/_redis.py",
    "litellm/_redis_credential_provider.py",
    "litellm/caching/redis_batch.py",
    "litellm/caching/redis_cluster_node_isolation.py",
    "litellm/caching/valkey_semantic_cache.py",
    "litellm/llms/valkey/vector_stores/transformation.py",
    "litellm/proxy/db/db_transaction_queue/redis_update_buffer.py",
)


@pytest.fixture
def lint_root(tmp_path: Path) -> Path:
    for name in ("ruff.toml", "ruff-strict.toml"):
        shutil.copyfile(_REPO_ROOT / name, tmp_path / name)
    return tmp_path


def _write_source(root: Path, filename: str, source: str) -> Path:
    path: Final = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _ruff(root: Path, target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            "-I",
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--config",
            str(root / "ruff-strict.toml"),
            "--output-format",
            "concise",
            str(target),
        ),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _diagnostic_codes(output: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in re.finditer(r": ([A-Z]+\d+) ", output))


@pytest.mark.parametrize("filename", _CALLERS)
@pytest.mark.parametrize("source", _REDIS_IMPORTS)
def test_raw_redis_import_is_rejected_outside_exceptions(lint_root: Path, filename: str, source: str) -> None:
    path: Final = _write_source(lint_root, filename, source)
    result: Final = _ruff(lint_root, path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _diagnostic_codes(result.stdout) == ("TID251",), result.stdout
    assert "Reach Redis through litellm.redis" in result.stdout


@pytest.mark.parametrize("filename", _EXCEPTIONS)
@pytest.mark.parametrize("source", _REDIS_IMPORTS)
def test_raw_redis_import_is_accepted_in_existing_exception(lint_root: Path, filename: str, source: str) -> None:
    path: Final = _write_source(lint_root, filename, source)
    result: Final = _ruff(lint_root, path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("filename", _CALLERS)
@pytest.mark.parametrize(
    "source",
    (
        "import litellm.redis\n\ndir(litellm.redis)\n",
        "import litellm.redis as redis_store\n\ndir(redis_store)\n",
        "from litellm import redis\n\ndir(redis)\n",
        "from litellm.redis.stores.pod_lock import PodLockStore\n\ndir(PodLockStore)\n",
        "from litellm.redis.transport.contract import RedisTransport\n\ndir(RedisTransport)\n",
        "from litellm.redis.transport.redis_py import RedisPyTransport\n\ndir(RedisPyTransport)\n",
    ),
)
def test_store_and_transport_imports_remain_allowed(lint_root: Path, filename: str, source: str) -> None:
    path: Final = _write_source(lint_root, filename, source)
    result: Final = _ruff(lint_root, path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_directory_discovery_rejects_raw_redis_import(lint_root: Path) -> None:
    _write_source(lint_root, "litellm/proxy/_redis_boundary_probe.py", "import redis\n\nredis.Redis()\n")
    result: Final = _ruff(lint_root, lint_root / "litellm")
    assert result.returncode == 1, result.stdout + result.stderr
    assert _diagnostic_codes(result.stdout) == ("TID251",), result.stdout
    assert "litellm/proxy/_redis_boundary_probe.py:1:8:" in result.stdout
