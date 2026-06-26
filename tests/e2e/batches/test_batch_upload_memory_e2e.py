"""Faithful memory/OOM regression guard for the batch-upload transform (#31036).

LIT-3382: the OpenAI->Vertex batch JSONL upload transform buffered the whole file
and made several in-memory copies, so a large upload OOM-killed the pod. #31036
streams the upload so memory stays bounded regardless of file size. This guard
uploads a large batch JSONL through the vertex_ai managed-files path while sampling
the proxy's memory, and asserts peak growth stays within a small multiple of the
file size: the streaming fix passes, the buffering regression (memory scaling with
file size) blows past the bound.

Gated off by default. The vertex_ai files handler stages the JSONL in GCS, which
needs a writable bucket on a billing-enabled project, and a meaningful memory
assertion needs real memory headroom (a ~3.8 GB local docker VM OOMs the whole
proxy before the signal is clean). Both hold on EKS, not on the local
docker-compose stack. To enable: configure VERTEXAI creds + a GCS bucket
(`gcs_bucket_name`) on the vertex deployment, set E2E_BATCH_MEMORY_ENABLED=1, and
inject the MemorySampler for the environment (the docker cgroup sampler here, or a
pod-based one on EKS).

NOTE: not yet validated against a live streaming vertex path (billing-blocked in
the dev environment). Treat MAX_GROWTH_RATIO as a starting point and confirm the
pass/fail margin when first enabling it.
"""

from __future__ import annotations

import os

import pytest

from batch_client import BatchFilesClient, batch_jsonl
from lifecycle import ResourceManager
from memory import DockerCgroupSampler, MemorySampler

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("E2E_BATCH_MEMORY_ENABLED") != "1",
        reason="needs the streaming vertex_ai files path: GCS bucket + billing + "
        "memory headroom (enable on EKS with E2E_BATCH_MEMORY_ENABLED=1)",
    ),
]

VERTEX_MODEL = os.getenv("E2E_BATCH_MEMORY_MODEL", "gemini-2.5-flash-vertex")
FILE_BYTES = int(os.getenv("E2E_BATCH_MEMORY_FILE_MB", "200")) * 1024 * 1024
MAX_GROWTH_RATIO = float(os.getenv("E2E_BATCH_MEMORY_MAX_RATIO", "3.0"))
CONTAINER = os.getenv("E2E_LITELLM_CONTAINER", "e2e-litellm-1")


@pytest.fixture
def sampler() -> MemorySampler:
    return DockerCgroupSampler(container=CONTAINER)


def _sized_batch(model: str, target_bytes: int, *, pad_bytes: int = 2000) -> bytes:
    per_line = len(batch_jsonl(model, lines=1, pad_bytes=pad_bytes))
    lines = max(1, target_bytes // per_line)
    return batch_jsonl(model, lines=lines, pad_bytes=pad_bytes)


def test_large_vertex_batch_upload_memory_is_bounded(
    client: BatchFilesClient,
    resources: ResourceManager,
    scoped_key: str,
    sampler: MemorySampler,
) -> None:
    content = _sized_batch(VERTEX_MODEL, FILE_BYTES)
    file_size = len(content)

    uploaded, mem = sampler.measure(
        lambda: client.upload_batch_file(scoped_key, content, target_model=VERTEX_MODEL)
    )
    resources.defer(lambda: client.delete_file(uploaded.id))

    assert uploaded.status == "uploaded"
    ratio = mem.growth_bytes / file_size
    assert ratio < MAX_GROWTH_RATIO, (
        f"proxy memory grew {mem.growth_bytes / 1e6:.0f}MB for a "
        f"{file_size / 1e6:.0f}MB upload (ratio {ratio:.1f}x >= {MAX_GROWTH_RATIO}x); "
        f"the streaming transform must keep upload memory bounded (#31036 OOM regression)"
    )
