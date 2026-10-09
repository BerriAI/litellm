"""Tests for the legacy Redis rate-limit key maintenance command."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cleanup_legacy_rate_limit_keys.py"
_spec = importlib.util.spec_from_file_location("cleanup_legacy_rate_limit_keys", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
cleanup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cleanup
_spec.loader.exec_module(cleanup)


class FakeRedis:
    def __init__(self, keys: list[str], ttls: dict[str, int]):
        self.keys = keys
        self.ttls = ttls
        self.scan_calls: list[tuple[str, int]] = []
        self.unlink_calls: list[str] = []

    def scan_iter(self, *, match: str, count: int):
        self.scan_calls.append((match, count))
        yield from self.keys

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -2)

    def unlink(self, key: str) -> int:
        self.unlink_calls.append(key)
        return 1


class DeleteOnlyRedis(FakeRedis):
    unlink = None

    def __init__(self, keys: list[str], ttls: dict[str, int]):
        super().__init__(keys, ttls)
        self.delete_calls: list[str] = []

    def delete(self, key: str) -> int:
        self.delete_calls.append(key)
        return 1


def test_legacy_key_kind_distinguishes_old_formats_from_v2() -> None:
    assert cleanup.legacy_key_kind("global_router:id:model:tpm:13-07") == "tpm"
    assert cleanup.legacy_key_kind("model:rpm:13-07") == "rpm"
    assert cleanup.legacy_key_kind("13-07:model") == "dynamic"
    assert cleanup.legacy_key_kind("global_router:id:model:tpm:v2:29300000") is None
    assert cleanup.legacy_key_kind("v2:29300000:model") is None
    assert cleanup.legacy_key_kind("global_router:id:model:rpm:24-00") is None


def test_namespace_is_required_for_matching_when_supplied() -> None:
    key = "tenant-a:global_router:id:model:rpm:13-07"
    assert cleanup.legacy_key_kind(key, "tenant-a") == "rpm"
    assert cleanup.legacy_key_kind(key, "tenant-b") is None
    with pytest.raises(ValueError, match="whitespace"):
        cleanup.legacy_key_kind(key, "tenant a")


def test_dry_run_scans_cluster_aware_iterator_without_deleting() -> None:
    keys = [
        "tenant-a:global_router:id:model:tpm:13-07",
        "tenant-a:model:rpm:13-07",
        "tenant-a:13-07:model",
        "tenant-a:global_router:id:model:tpm:v2:29300000",
        "tenant-b:global_router:id:model:rpm:13-07",
    ]
    redis = FakeRedis(keys, {keys[0]: -1, keys[1]: 30, keys[2]: 4})

    report = cleanup.cleanup_legacy_keys(redis, namespace="tenant-a", count=17)

    assert redis.scan_calls == [("tenant-a:*", 17)]
    assert report.scanned == 5
    assert report.candidates == 3
    assert report.permanent == 1
    assert report.finite == 2
    assert report.by_kind == {"dynamic": 1, "rpm": 1, "tpm": 1}
    assert report.deleted == 0
    assert redis.unlink_calls == []


def test_apply_requires_namespace() -> None:
    old_key = "global_router:id:model:rpm:13-07"
    redis = FakeRedis([old_key, old_key, "global_router:id:model:rpm:v2:29300000"], {old_key: -1})

    with pytest.raises(ValueError, match="namespace"):
        cleanup.cleanup_legacy_keys(redis, apply=True)

    assert redis.unlink_calls == []


def test_apply_deletes_only_unique_permanent_legacy_keys() -> None:
    permanent_key = "tenant-a:global_router:id:model:rpm:13-07"
    finite_key = "tenant-a:global_router:id:model:tpm:13-07"
    redis = FakeRedis(
        [permanent_key, permanent_key, finite_key, "tenant-a:global_router:id:model:rpm:v2:29300000"],
        {permanent_key: -1, finite_key: 30},
    )

    report = cleanup.cleanup_legacy_keys(redis, namespace="tenant-a", apply=True)

    assert report.candidates == 2
    assert report.permanent == 1
    assert report.finite == 1
    assert report.deleted == 1
    assert report.failed == 0
    assert redis.unlink_calls == [permanent_key]


def test_apply_falls_back_to_single_key_delete() -> None:
    old_key = "tenant-a:13-07:model"
    redis = DeleteOnlyRedis([old_key], {old_key: 9})

    report = cleanup.cleanup_legacy_keys(redis, namespace="tenant-a", apply=True)

    assert report.deleted == 0
    assert redis.delete_calls == []


def test_apply_falls_back_to_single_key_delete_for_permanent_key() -> None:
    old_key = "tenant-a:13-07:model"
    redis = DeleteOnlyRedis([old_key], {old_key: -1})

    report = cleanup.cleanup_legacy_keys(redis, namespace="tenant-a", apply=True)

    assert report.deleted == 1
    assert redis.delete_calls == [old_key]


def test_connection_overrides_require_an_explicit_host_and_port() -> None:
    with pytest.raises(ValueError, match="--host"):
        cleanup._connection_kwargs(Namespace(host=None, port=6380, db=None))
    with pytest.raises(ValueError, match="--port"):
        cleanup._connection_kwargs(Namespace(host="redis.example", port=None, db=None))


def test_connection_overrides_do_not_accept_a_url() -> None:
    kwargs = cleanup._connection_kwargs(Namespace(host="redis.example", port=6380, db=2))

    assert kwargs == {"host": "redis.example", "port": 6380, "db": 2}


def test_connection_overrides_do_not_mix_with_cluster_configuration(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_CLUSTER_NODES", '[{"host": "cluster.example", "port": 6379}]')

    with pytest.raises(ValueError, match="Cluster or Sentinel"):
        cleanup._connect(Namespace(host="redis.example", port=6380, db=None))


def test_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        cleanup.cleanup_legacy_keys(FakeRedis([], {}), count=0)
