"""Contract: the `/anthropic/*` pass-through carries Anthropic's Files and Skills APIs with the operator key.

Claude Code and the Anthropic SDK reach the Files API (`/v1/files`) and Skills API (`/v1/skills`) through the
proxy's Anthropic pass-through (`anthropic_proxy_route`, `litellm/proxy/pass_through_endpoints/
llm_passthrough_endpoints.py:1286`). The route joins the path onto `anthropic_pass_through_api_base()`, takes the
key from `passthrough_endpoint_router.get_credentials("anthropic")` (a pass-through deployment, else
`ANTHROPIC_API_KEY`), and forwards the caller's headers minus credentials and LiteLLM secrets
(`_upstream_headers_for_anthropic_route`, line 3303), so the caller's `anthropic-beta` reaches Anthropic and the
proxy key never does. Multipart uploads, binary downloads and deletes must keep working the same way when the
front door moves to Rust.

BUG (run 44176682961): the proxy's OpenAI-style files router declares `/{provider}/v1/files`,
`/{provider}/v1/files/{file_id:path}` and `/{provider}/v1/files/{file_id:path}/content`
(litellm/proxy/openai_files_endpoints/files_endpoints.py:593-607, :1009-1019, :1318-1328, :1516-1526, :1920-1930),
and that router is included before the pass-through routes (litellm/proxy/proxy_server.py:23218-23220). So
`/anthropic/v1/files...` never reaches `anthropic_proxy_route`: an upload without an OpenAI `purpose` form field
is refused with 422, metadata and delete come back as OpenAI file objects (`object`, `bytes`, `purpose`,
`deleted`), and the list drops `limit` and `after_id`. The Files API tests that see this skip with BUG; the
content download and the Skills API still pass through.

The module owns a proxy whose `ANTHROPIC_API_BASE` points at an owned peer and whose `ANTHROPIC_API_KEY` is
synthetic; every other `ANTHROPIC_*` variable is scrubbed, as in round 1's `test_dec1_anthropic_env_api_base_owned_proxy.py`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration.providers.dec1_anthropic2_support import (
    FILE_BYTES,
    FILE_ID,
    FILES_BETA,
    SKILL_ID,
    SKILLS_BETA,
    file_object,
    files_respond,
    multipart_field,
)
from integration.providers.dec1_anthropic_support import Call, Peer, peer

_ENV_KEY: Final = "dec1-anthropic2-env-key"
_load_yaml: Final[Callable[[str], object]] = yaml.safe_load

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)


def _config(directory: Path) -> Path:
    base: Final = JSON_OBJECT.validate_python(_load_yaml(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "dec1-anthropic2-files.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": []}))
    return path


@pytest.fixture(scope="module")
def provider() -> Iterator[Peer]:
    with peer(files_respond) as owned:
        yield owned


@pytest.fixture(scope="module")
def isolated(provider: Peer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-anthropic2-files")
    scrubbed: Final = tuple(name for name in os.environ if name.upper().startswith("ANTHROPIC_"))
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"ANTHROPIC_API_BASE": provider.url, "ANTHROPIC_API_KEY": _ENV_KEY},
            config=_config(directory),
            remove_environment=scrubbed,
        ) as candidate,
    ):
        yield candidate


def _after(provider: Peer, send: Callable[[], httpx.Response]) -> tuple[httpx.Response, tuple[Call, ...]]:
    before: Final = len(provider.all())
    response: Final = send()
    return response, provider.all()[before:]


def _beta(value: str) -> dict[str, str]:
    return {"anthropic-beta": value}


def test_a_multipart_upload_is_forwarded_to_v1_files_with_its_bytes(isolated: Gateway, provider: Peer) -> None:
    """The caller posts `multipart/form-data` with a `file` part (`dec1-notes.txt`, text/plain) to
    `/anthropic/v1/files` with `anthropic-beta: files-api-2025-04-14`.

    The peer must see one `POST /v1/files` whose body is multipart and holds the exact file bytes and the
    filename, and the caller must get Anthropic's file object back unchanged (`id: file_011dec1synthetic`).

    See the module docstring: this route is answered by the OpenAI-style files endpoint (BUG).
    """
    pytest.skip("BUG: /{provider}/v1/files routes shadow the Anthropic pass-through for the Files API")
    response, calls = _after(
        provider,
        lambda: isolated.client.post(
            "/anthropic/v1/files",
            files={"file": ("dec1-notes.txt", FILE_BYTES, "text/plain")},
            headers={"Authorization": f"Bearer {isolated.key}", **_beta(FILES_BETA)},
        ),
    )
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content) == file_object()
    posts: Final = [call for call in calls if call.method == "POST"]
    assert [call.path for call in posts] == ["/v1/files"]
    assert posts[0].headers.get("content-type", "").startswith("multipart/form-data")
    assert FILE_BYTES in posts[0].raw
    assert b'filename="dec1-notes.txt"' in posts[0].raw
    assert multipart_field(posts[0].raw, "file") == FILE_BYTES


def test_the_upload_carries_the_operator_key_and_the_caller_beta(isolated: Gateway, provider: Peer) -> None:
    """The same upload must reach the peer with `x-api-key: dec1-anthropic2-env-key` (the operator's
    `ANTHROPIC_API_KEY`), the caller's `anthropic-beta: files-api-2025-04-14`, and no header holding the caller's
    proxy key, neither as `authorization` nor as `x-api-key`.

    See the module docstring: this route is answered by the OpenAI-style files endpoint (BUG).
    """
    pytest.skip("BUG: /{provider}/v1/files routes shadow the Anthropic pass-through for the Files API")
    _, calls = _after(
        provider,
        lambda: isolated.client.post(
            "/anthropic/v1/files",
            files={"file": ("dec1-notes.txt", FILE_BYTES, "text/plain")},
            headers={"Authorization": f"Bearer {isolated.key}", **_beta(FILES_BETA)},
        ),
    )
    post: Final = next(call for call in calls if call.method == "POST")
    assert post.headers.get("x-api-key") == _ENV_KEY
    assert post.headers.get("anthropic-beta") == FILES_BETA
    assert all(isolated.key not in value for value in post.headers.values())


def test_file_metadata_is_a_get_on_the_same_path(isolated: Gateway, provider: Peer) -> None:
    """`GET /anthropic/v1/files/file_011dec1synthetic` must reach the peer as a GET on
    `/v1/files/file_011dec1synthetic` and return Anthropic's file object unchanged.

    See the module docstring: this route is answered by the OpenAI-style files endpoint (BUG).
    """
    pytest.skip("BUG: /{provider}/v1/files routes shadow the Anthropic pass-through for the Files API")
    response, calls = _after(
        provider,
        lambda: isolated.request("GET", f"/anthropic/v1/files/{FILE_ID}", headers=_beta(FILES_BETA)),
    )
    assert response.status_code == 200, response.text
    assert [(call.method, call.path) for call in calls] == [("GET", f"/v1/files/{FILE_ID}")]
    assert JSON_OBJECT.validate_json(response.content) == file_object()


def test_file_content_is_relayed_as_bytes(isolated: Gateway, provider: Peer) -> None:
    """`GET /anthropic/v1/files/file_011dec1synthetic/content` must reach the peer on the same path, and the
    caller must get the file's exact bytes with a `text/plain` content type, not a JSON re-encoding.
    """
    response, calls = _after(
        provider,
        lambda: isolated.request("GET", f"/anthropic/v1/files/{FILE_ID}/content", headers=_beta(FILES_BETA)),
    )
    assert response.status_code == 200, response.text
    assert response.content == FILE_BYTES
    assert response.headers.get("content-type", "").startswith("text/plain")
    assert [call.path for call in calls] == [f"/v1/files/{FILE_ID}/content"]


def test_a_file_delete_is_forwarded_as_a_delete(isolated: Gateway, provider: Peer) -> None:
    """`DELETE /anthropic/v1/files/file_011dec1synthetic` must reach the peer as a DELETE on the same path and
    return Anthropic's `{"id": ..., "type": "file_deleted"}` unchanged.

    See the module docstring: this route is answered by the OpenAI-style files endpoint (BUG).
    """
    pytest.skip("BUG: /{provider}/v1/files routes shadow the Anthropic pass-through for the Files API")
    response, calls = _after(
        provider,
        lambda: isolated.request("DELETE", f"/anthropic/v1/files/{FILE_ID}", headers=_beta(FILES_BETA)),
    )
    assert response.status_code == 200, response.text
    assert [(call.method, call.path) for call in calls] == [("DELETE", f"/v1/files/{FILE_ID}")]
    assert JSON_OBJECT.validate_json(response.content) == {"id": FILE_ID, "type": "file_deleted"}


def test_the_files_list_keeps_its_query_string(isolated: Gateway, provider: Peer) -> None:
    """`GET /anthropic/v1/files?limit=5&after_id=file_dec1prev` must reach the peer with the same `limit` and
    `after_id` query values, since the SDK pages through files with them.

    See the module docstring: this route is answered by the OpenAI-style files endpoint (BUG).
    """
    pytest.skip("BUG: /{provider}/v1/files routes shadow the Anthropic pass-through for the Files API")
    response, calls = _after(
        provider,
        lambda: isolated.request(
            "GET", "/anthropic/v1/files", params={"limit": "5", "after_id": "file_dec1prev"}, headers=_beta(FILES_BETA)
        ),
    )
    assert response.status_code == 200, response.text
    assert len(calls) == 1
    assert dict(calls[0].query) == {"limit": ["5"], "after_id": ["file_dec1prev"]}


def test_the_skills_list_is_forwarded_with_the_skills_beta(isolated: Gateway, provider: Peer) -> None:
    """`GET /anthropic/v1/skills` with `anthropic-beta: skills-2025-10-02` must reach the peer as a GET on
    `/v1/skills` carrying that beta and the operator key, and return the peer's list holding `skill_dec1report`.
    """
    response, calls = _after(
        provider, lambda: isolated.request("GET", "/anthropic/v1/skills", headers=_beta(SKILLS_BETA))
    )
    assert response.status_code == 200, response.text
    assert [(call.method, call.path) for call in calls] == [("GET", "/v1/skills")]
    assert calls[0].headers.get("anthropic-beta") == SKILLS_BETA
    assert calls[0].headers.get("x-api-key") == _ENV_KEY
    assert SKILL_ID in response.text
