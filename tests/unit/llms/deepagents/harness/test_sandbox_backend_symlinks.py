"""A repository must not be able to reach host files through symlinks, in any file tool."""

import asyncio
import os
from pathlib import Path

import pytest

from litellm.harness.sandbox.local import LocalSandbox

backend = pytest.importorskip("litellm.llms.deepagents.harness.sandbox_backend")

SECRET = "AWS_SECRET_ACCESS_KEY=leaked-from-host"


@pytest.fixture
def repo_with_escape_links(tmp_path: Path) -> Path:
    host = tmp_path / "host_home"
    host.mkdir()
    (host / "credentials").write_text(SECRET + "\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello\n")
    os.symlink(host / "credentials", repo / "creds_link")
    os.symlink(host, repo / "home_link")
    return repo


async def _backend(repo: Path) -> object:
    return backend.SandboxBackend(LocalSandbox(str(repo)), loop=asyncio.get_running_loop(), writable=False)


async def test_grep_whole_repo_skips_symlinks_out_of_workspace(repo_with_escape_links: Path) -> None:
    b = await _backend(repo_with_escape_links)
    result = await b.agrep("AWS_SECRET")
    assert not result.matches, f"grep followed a symlink out of the repo: {result}"


async def test_grep_rooted_at_symlink_dir_is_refused(repo_with_escape_links: Path) -> None:
    b = await _backend(repo_with_escape_links)
    result = await b.agrep("AWS_SECRET", path="/home_link")
    assert result.error and "outside the workspace" in result.error
    assert not result.matches


async def test_read_through_symlink_is_refused(repo_with_escape_links: Path) -> None:
    b = await _backend(repo_with_escape_links)
    result = await b.aread("/creds_link")
    assert result.error and "outside the workspace" in result.error
    assert SECRET not in str(result.file_data)


async def test_glob_does_not_list_files_behind_symlinks(repo_with_escape_links: Path) -> None:
    b = await _backend(repo_with_escape_links)
    result = await b.aglob("**/*")
    paths = [m["path"] for m in result.matches or []]
    assert paths == ["/README.md"]


async def test_grep_still_finds_real_repo_files(repo_with_escape_links: Path) -> None:
    b = await _backend(repo_with_escape_links)
    result = await b.agrep("hello")
    assert [(m["path"], m["line"]) for m in result.matches] == [("/README.md", 1)]
