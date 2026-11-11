"""Sandboxes for litellm.harness: where the runtime runs and which files it can touch."""

from litellm.harness.sandbox.base import CompletedRun, Process, Sandbox
from litellm.harness.sandbox.docker import DockerSandbox, docker
from litellm.harness.sandbox.local import LocalSandbox, local
from litellm.harness.sandbox.snapshot import (
    build_file_changes,
    capture_text_contents,
    diff_snapshots,
    snapshot_local,
)

__all__ = (
    "CompletedRun",
    "DockerSandbox",
    "LocalSandbox",
    "Process",
    "Sandbox",
    "build_file_changes",
    "capture_text_contents",
    "diff_snapshots",
    "docker",
    "local",
    "snapshot_local",
)
