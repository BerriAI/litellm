"""Unit tests for fargate registry module.

Mocks boto3 ECR client + subprocess. No real AWS / docker calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from litellm.proxy.managed_agents_endpoints.fargate import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": code}},
        operation_name="op",
    )


@pytest.fixture
def mock_ecr():
    return MagicMock()


# ---------------------------------------------------------------------------
# compute_dockerfile_hash
# ---------------------------------------------------------------------------


def test_compute_dockerfile_hash_deterministic(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.11\nRUN pip install foo\n")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "app.py").write_text("print('hi')\n")

    h1 = registry.compute_dockerfile_hash(str(df), str(ctx))
    h2 = registry.compute_dockerfile_hash(str(df), str(ctx))

    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_compute_dockerfile_hash_changes_on_dockerfile_change(tmp_path):
    df = tmp_path / "Dockerfile"
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "app.py").write_text("print('hi')\n")

    df.write_text("FROM python:3.11\n")
    h1 = registry.compute_dockerfile_hash(str(df), str(ctx))

    df.write_text("FROM python:3.12\n")
    h2 = registry.compute_dockerfile_hash(str(df), str(ctx))

    assert h1 != h2


def test_compute_dockerfile_hash_changes_on_context_change(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.11\n")

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "app.py").write_text("print('hi')\n")

    h1 = registry.compute_dockerfile_hash(str(df), str(ctx))

    # Add a new file → hash should differ
    (ctx / "extra.py").write_text("print('extra')\n")
    h2 = registry.compute_dockerfile_hash(str(df), str(ctx))

    assert h1 != h2


# ---------------------------------------------------------------------------
# image_exists
# ---------------------------------------------------------------------------


def test_image_exists_returns_true_on_found(mock_ecr):
    mock_ecr.describe_images.return_value = {"imageDetails": [{"imageTags": ["abc"]}]}

    with patch.object(registry, "_ecr", return_value=mock_ecr):
        assert registry.image_exists("us-west-2", "repo", "abc") is True


def test_image_exists_returns_false_on_not_found(mock_ecr):
    mock_ecr.describe_images.side_effect = _client_error("ImageNotFoundException")

    with patch.object(registry, "_ecr", return_value=mock_ecr):
        assert registry.image_exists("us-west-2", "repo", "abc") is False


# ---------------------------------------------------------------------------
# ensure_ecr_repo
# ---------------------------------------------------------------------------


def test_ensure_ecr_repo_creates_when_missing(mock_ecr):
    uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/newrepo"
    mock_ecr.describe_repositories.side_effect = _client_error(
        "RepositoryNotFoundException"
    )
    mock_ecr.create_repository.return_value = {"repository": {"repositoryUri": uri}}

    with patch.object(registry, "_ecr", return_value=mock_ecr):
        result = registry.ensure_ecr_repo("us-west-2", "newrepo")

    assert result == uri
    assert ".dkr.ecr.us-west-2.amazonaws.com/newrepo" in result
    mock_ecr.create_repository.assert_called_once()


# ---------------------------------------------------------------------------
# build_and_push orchestration
# ---------------------------------------------------------------------------


def test_build_and_push_cache_hit_skips_build(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    ctx = tmp_path / "ctx"
    ctx.mkdir()

    repo_uri = "123.dkr.ecr.us-west-2.amazonaws.com/repo"

    with (
        patch.object(registry, "ensure_ecr_repo", return_value=repo_uri),
        patch.object(registry, "image_exists", return_value=True),
        patch.object(registry, "docker_login") as login_mock,
        patch.object(registry, "docker_build") as build_mock,
        patch.object(registry, "docker_push") as push_mock,
        patch.object(registry.subprocess, "Popen") as popen_mock,
        patch.object(registry.subprocess, "run") as run_mock,
    ):
        result = registry.build_and_push(
            region="us-west-2",
            repo_name="repo",
            dockerfile_path=str(df),
            context_dir=str(ctx),
            content_hash="abc123",
        )

    assert result == f"{repo_uri}:abc123"
    login_mock.assert_not_called()
    build_mock.assert_not_called()
    push_mock.assert_not_called()
    popen_mock.assert_not_called()
    run_mock.assert_not_called()


def test_build_and_push_cache_miss_builds_and_pushes(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    ctx = tmp_path / "ctx"
    ctx.mkdir()

    repo_uri = "123.dkr.ecr.us-west-2.amazonaws.com/repo"
    call_order: list = []

    with (
        patch.object(registry, "ensure_ecr_repo", return_value=repo_uri),
        patch.object(registry, "image_exists", return_value=False),
        patch.object(
            registry,
            "docker_login",
            side_effect=lambda *a, **kw: call_order.append("login"),
        ) as login_mock,
        patch.object(
            registry,
            "docker_build",
            side_effect=lambda *a, **kw: call_order.append("build"),
        ) as build_mock,
        patch.object(
            registry,
            "docker_push",
            side_effect=lambda *a, **kw: call_order.append("push"),
        ) as push_mock,
    ):
        result = registry.build_and_push(
            region="us-west-2",
            repo_name="repo",
            dockerfile_path=str(df),
            context_dir=str(ctx),
            content_hash="abc123",
        )

    assert result == f"{repo_uri}:abc123"
    assert call_order == ["login", "build", "push"]
    login_mock.assert_called_once()
    build_mock.assert_called_once()
    push_mock.assert_called_once()
