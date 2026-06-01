"""ECR repo lifecycle + docker build/push mechanics for managed agent images."""

import base64
import hashlib
import os
import subprocess
from typing import Any, Callable, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from litellm._logging import verbose_proxy_logger

_BUILD_TIMEOUT_SECONDS = 30 * 60
_PUSH_TIMEOUT_SECONDS = 30 * 60

_clients: Dict[str, Any] = {}


def _ecr(region: str):
    key = f"ecr:{region}"
    if key not in _clients:
        _clients[key] = boto3.client("ecr", region_name=region)
    return _clients[key]


def ensure_ecr_repo(region: str, repo_name: str) -> str:
    client = _ecr(region)
    try:
        r = client.describe_repositories(repositoryNames=[repo_name])
        return r["repositories"][0]["repositoryUri"]
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "RepositoryNotFoundException":
            raise

    try:
        r = client.create_repository(
            repositoryName=repo_name, imageScanningConfiguration={"scanOnPush": True}
        )
        return r["repository"]["repositoryUri"]
    except ClientError as e:
        if (
            e.response.get("Error", {}).get("Code")
            != "RepositoryAlreadyExistsException"
        ):
            raise
        r = client.describe_repositories(repositoryNames=[repo_name])
        return r["repositories"][0]["repositoryUri"]


def image_exists(region: str, repo_name: str, tag: str) -> bool:
    client = _ecr(region)
    try:
        client.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ImageNotFoundException":
            return False
        raise


def compute_dockerfile_hash(dockerfile_path: str, context_dir: Optional[str]) -> str:
    h = hashlib.sha256()

    with open(dockerfile_path, "rb") as f:
        h.update(b"dockerfile:")
        h.update(f.read())

    ctx = (
        context_dir
        if context_dir is not None
        else os.path.dirname(os.path.abspath(dockerfile_path))
    )
    if ctx and os.path.isdir(ctx):
        entries = []
        for root, dirs, files in os.walk(ctx):
            dirs.sort()
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, ctx)
                entries.append((rel, full))
        entries.sort(key=lambda e: e[0])
        for rel, full in entries:
            h.update(b"\x00path:")
            h.update(rel.encode("utf-8"))
            try:
                with open(full, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
            except OSError:
                continue

    return h.hexdigest()


def docker_login(region: str) -> None:
    client = _ecr(region)
    r = client.get_authorization_token()
    auth = r["authorizationData"][0]
    token = auth["authorizationToken"]
    registry = auth["proxyEndpoint"]
    decoded = base64.b64decode(token).decode("utf-8")
    user, _, password = decoded.partition(":")
    if not user or not password:
        raise RuntimeError("ECR get_authorization_token returned malformed credentials")

    try:
        result = subprocess.run(
            ["docker", "login", "-u", user, "--password-stdin", registry],
            input=password,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError("docker daemon required on proxy host")

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"docker login failed: {tail}")


def _stream_subprocess(
    cmd: list,
    *,
    timeout: int,
    log_callback: Optional[Callable[[str], None]],
    op_name: str,
) -> None:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        raise RuntimeError("docker daemon required on proxy host")

    stderr_tail: list = []
    stdout_lines: list = []

    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip("\n")
                stdout_lines.append(line)
                if log_callback is not None:
                    try:
                        log_callback(line)
                    except Exception:
                        pass
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(f"{op_name} timed out after {timeout}s")

    if proc.stderr is not None:
        try:
            stderr_tail.append(proc.stderr.read())
        except Exception:
            pass

    if proc.returncode != 0:
        tail = "".join(stderr_tail).strip()
        if not tail:
            tail = "\n".join(stdout_lines[-50:])
        tail = tail[-4000:]
        raise RuntimeError(f"{op_name} failed (exit {proc.returncode}): {tail}")


def docker_build(
    dockerfile_path: str,
    context_dir: str,
    image_uri: str,
    *,
    platform: str = "linux/amd64",
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    if not os.path.isfile(dockerfile_path):
        raise RuntimeError(f"dockerfile not found: {dockerfile_path}")
    ctx = (
        context_dir
        if context_dir
        else os.path.dirname(os.path.abspath(dockerfile_path))
    )
    if not os.path.isdir(ctx):
        raise RuntimeError(f"context dir not found: {ctx}")

    cmd = [
        "docker",
        "build",
        "--platform",
        platform,
        "-f",
        dockerfile_path,
        "-t",
        image_uri,
        ctx,
    ]
    _stream_subprocess(
        cmd,
        timeout=_BUILD_TIMEOUT_SECONDS,
        log_callback=log_callback,
        op_name="docker build",
    )


def docker_push(
    image_uri: str,
    *,
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    cmd = ["docker", "push", image_uri]
    _stream_subprocess(
        cmd,
        timeout=_PUSH_TIMEOUT_SECONDS,
        log_callback=log_callback,
        op_name="docker push",
    )


def build_and_push(
    *,
    region: str,
    repo_name: str,
    dockerfile_path: str,
    context_dir: str,
    content_hash: str,
    platform: str = "linux/amd64",
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    if not os.path.isfile(dockerfile_path):
        raise RuntimeError(f"dockerfile not found: {dockerfile_path}")
    ctx = (
        context_dir
        if context_dir
        else os.path.dirname(os.path.abspath(dockerfile_path))
    )
    if not os.path.isdir(ctx):
        raise RuntimeError(f"context dir not found: {ctx}")

    repo_uri = ensure_ecr_repo(region, repo_name)
    tag = content_hash
    image_uri = f"{repo_uri}:{tag}"

    if image_exists(region, repo_name, tag):
        verbose_proxy_logger.info(
            f"ECR cache hit for {repo_name}:{tag} — skipping build"
        )
        return image_uri

    verbose_proxy_logger.info(
        f"Building image {repo_name}:{tag} from {dockerfile_path} "
        f"(context={ctx}, platform={platform})"
    )
    docker_login(region)
    docker_build(
        dockerfile_path,
        ctx,
        image_uri,
        platform=platform,
        log_callback=log_callback,
    )
    docker_push(image_uri, log_callback=log_callback)
    verbose_proxy_logger.info(f"Pushed image {image_uri}")
    return image_uri
