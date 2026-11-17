"""Contract: managed batch files and batches, who may see them, and how batch JSONL is rewritten on upload.

Ownership (`litellm/llms/base_llm/managed_resources/isolation.py:110` `can_access_resource`): a proxy admin
sees everything; otherwise a caller sees a managed file or batch when its row's `created_by` is the caller's
owner id, or when the row's `team_id` is the caller's team. A managed id with no row is 404 (`File not found:
<id>` for files, `Object not found: <id>` for batches); a row the caller may not see is 403. The same check
guards a raw provider file id once it is recorded in `flat_model_file_ids` (#51498), so the provider spelling
is no way around it.

Deletion (`managed_files.py` `_check_file_deletion_allowed`): while the batch poller is scheduled (the default
`PROXY_BATCH_POLLING_ENABLED=true`), deleting a managed file that a non-terminal, not yet billed batch uses
as input is 400 and names the batch.

Rewriting (#51843, landed 2026-11-16): on a model-routed upload (`model` form field) only the rows whose
`body.model` is the alias are rewritten to the deployment's provider model, as compact UTF-8 JSON; every other
row reaches the provider byte for byte, and a file with no alias row is sent unchanged under its own name.

Assumed: `dec1_resources_support.booted_rig` (one owned proxy, scripted OpenAI-compatible upstream that
names an uploaded file `file-<filename stem>`, and `file-<first custom_id>` for a re-uploaded
`modified_file.jsonl`).
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.wire import Request
from integration.providers.dec1_resources_support import (
    JSON_OBJECT,
    OPENAI_MODEL,
    Rig,
    batch_line,
    booted_rig,
    error_of,
    jsonl,
    multipart_field,
    multipart_filename,
    path_of,
    unique,
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(tmp_path_factory.mktemp("dec1-batchfiles-managed-access")) as booted:
        yield booted


def _unpadded(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _managed_file(rig: Rig, key: str, stem: str) -> str:
    return rig.uploaded_id(
        f"{stem}.jsonl", jsonl(batch_line(OPENAI_MODEL, custom_id=stem)), key=key, target_model_names=OPENAI_MODEL
    )


def _managed_batch(rig: Rig, key: str, file_id: str) -> str:
    created: Final = rig.proxy.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        key=key,
    )
    assert created.status_code == 200, created.text
    identity: Final = JSON_OBJECT.validate_json(created.content)["id"]
    assert isinstance(identity, str)
    return identity


def _uploads(rig: Rig) -> list[Request]:
    return [request for request in rig.drained() if request.method == "POST" and path_of(request) == "/v1/files"]


def test_an_unknown_managed_file_id_is_404_naming_the_id(rig: Rig) -> None:
    """A well-formed managed file id (base64url of `litellm_proxy:application/jsonl;unified_id,<uuid>;...`) that
    was never issued is retrieved by a virtual key.

    The answer must be 404 whose error message contains `File not found: <id>`, and nothing reaches the
    provider. A rewrite that decodes the id and forwards the embedded provider id would leak other tenants' files.
    """
    unified: Final = _unpadded(
        f"litellm_proxy:application/jsonl;unified_id,{uuid.uuid4()};target_model_names,{OPENAI_MODEL};"
        "llm_output_file_id,file-dec1-never;llm_output_file_model_id,dec1-model-id"
    )
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.wire.drain()
        response: Final = rig.proxy.request("GET", f"/v1/files/{unified}", key=key)
        assert response.status_code == 404, response.text
        assert f"File not found: {unified}" in response.text
        assert rig.drained() == ()


def test_an_unknown_managed_batch_id_is_404_naming_the_id(rig: Rig) -> None:
    """A well-formed managed batch id (base64url of `litellm_proxy;model_id:<x>;llm_batch_id:batch_<y>`) that
    was never issued is retrieved by a virtual key.

    The answer must be 404 whose message contains `Object not found: <id>`, and no `GET /v1/batches/...` reaches
    the provider.
    """
    unified: Final = _unpadded(f"litellm_proxy;model_id:dec1-model-id;llm_batch_id:{unique('batch_never')}")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.wire.drain()
        response: Final = rig.proxy.request("GET", f"/v1/batches/{unified}", key=key)
        assert response.status_code == 404, response.text
        assert f"Object not found: {unified}" in response.text
        assert rig.drained() == ()


def test_a_teammate_reads_and_downloads_a_managed_file_created_by_another_team_key(rig: Rig) -> None:
    """Two keys belong to the same team; the first uploads a managed batch file.

    The second key's `GET /v1/files/<id>` must be 200 with the same unified id and its
    `GET /v1/files/<id>/content` must return the uploaded bytes: team membership grants access through the
    row's `team_id`.
    """
    stem: Final = unique("team-shared-file")
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team()
        owner: Final = scenario.key(team_id=team)
        teammate: Final = scenario.key(team_id=team)
        file_id: Final = _managed_file(rig, owner, stem)
        retrieved: Final = rig.proxy.request("GET", f"/v1/files/{file_id}", key=teammate)
        assert retrieved.status_code == 200, retrieved.text
        assert JSON_OBJECT.validate_json(retrieved.content)["id"] == file_id
        content: Final = rig.proxy.request("GET", f"/v1/files/{file_id}/content", key=teammate)
        assert content.status_code == 200, content.text
        assert json.loads(content.content.decode().strip().split("\n")[0])["custom_id"] == stem


def test_a_teammate_retrieves_a_managed_batch_created_by_another_team_key(rig: Rig) -> None:
    """Two keys belong to the same team; the first creates a managed batch.

    The second key's `GET /v1/batches/<id>` must be 200 and carry the same unified batch id.
    """
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team()
        owner: Final = scenario.key(team_id=team)
        teammate: Final = scenario.key(team_id=team)
        batch_id: Final = _managed_batch(rig, owner, _managed_file(rig, owner, unique("team-shared-batch")))
        retrieved: Final = rig.proxy.request("GET", f"/v1/batches/{batch_id}", key=teammate)
        assert retrieved.status_code == 200, retrieved.text
        assert JSON_OBJECT.validate_json(retrieved.content)["id"] == batch_id


def test_a_key_on_another_team_gets_403_on_file_read_content_and_delete(rig: Rig) -> None:
    """A key on team A uploads a managed file; a key on team B tries to read, download and delete it.

    All three must be 403 with a message containing `does not have access to the file`, nothing reaches the
    provider, and the `LiteLLM_ManagedFileTable` row still exists afterwards.
    """
    with rig.proxy.scenario() as scenario:
        owner: Final = scenario.key(team_id=scenario.team())
        stranger: Final = scenario.key(team_id=scenario.team())
        file_id: Final = _managed_file(rig, owner, unique("team-private-file"))
        rig.wire.drain()
        for method, path in (
            ("GET", f"/v1/files/{file_id}"),
            ("GET", f"/v1/files/{file_id}/content"),
            ("DELETE", f"/v1/files/{file_id}"),
        ):
            response = rig.proxy.request(method, path, key=stranger)
            assert response.status_code == 403, (method, path, response.text)
            assert "does not have access to the file" in response.text
        assert rig.drained() == ()
        assert rig.proxy.rows(
            'SELECT unified_file_id FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s', (file_id,)
        ) == [{"unified_file_id": file_id}]


def test_the_master_key_retrieves_any_keys_managed_batch(rig: Rig) -> None:
    """A virtual key with no team creates a managed batch; the master key (proxy admin) retrieves it.

    The answer must be 200 with the same unified id: admins bypass the owner check.
    """
    with rig.proxy.scenario() as scenario:
        owner: Final = scenario.key()
        batch_id: Final = _managed_batch(rig, owner, _managed_file(rig, owner, unique("admin-view")))
        retrieved: Final = rig.proxy.request("GET", f"/v1/batches/{batch_id}")
        assert retrieved.status_code == 200, retrieved.text
        assert JSON_OBJECT.validate_json(retrieved.content)["id"] == batch_id


def test_the_raw_provider_id_of_a_managed_file_is_guarded_like_the_managed_id(rig: Rig) -> None:
    """A key uploads a managed file whose provider id is `file-<stem>`; another key (no team) asks for
    `GET /v1/files/file-<stem>?custom_llm_provider=openai`.

    The answer must be 403 and nothing reaches the provider (#51498 matches every spelling of a recorded
    provider id). The owner asking for the same raw id still gets 200.
    """
    stem: Final = unique("raw-spelling")
    with rig.proxy.scenario() as scenario:
        owner: Final = scenario.key()
        stranger: Final = scenario.key()
        _managed_file(rig, owner, stem)
        rig.wire.drain()
        params: Final = {"custom_llm_provider": "openai"}
        refused: Final = rig.proxy.request("GET", f"/v1/files/file-{stem}", key=stranger, params=params)
        assert refused.status_code == 403, refused.text
        assert rig.drained() == ()
        allowed: Final = rig.proxy.request("GET", f"/v1/files/file-{stem}", key=owner, params=params)
        assert allowed.status_code == 200, allowed.text


def test_a_managed_file_still_resolves_when_the_request_also_names_a_model(rig: Rig) -> None:
    """The owner retrieves its managed file with `?model=<the target model>` added (#51780).

    The answer must be 200 with the unified id, and the upstream must get `GET /v1/files/file-<stem>`: the
    unified id wins over model routing instead of being sent to the provider as if it were a raw id.
    """
    stem: Final = unique("with-model")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        file_id: Final = _managed_file(rig, key, stem)
        rig.wire.drain()
        response: Final = rig.proxy.request("GET", f"/v1/files/{file_id}", key=key, params={"model": OPENAI_MODEL})
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["id"] == file_id
        assert [path_of(request) for request in rig.drained() if request.method == "GET"] == [f"/v1/files/file-{stem}"]


def test_deleting_a_managed_file_used_by_an_unbilled_batch_is_400_naming_the_batch(rig: Rig) -> None:
    """A key uploads a managed file and creates a batch from it (status `validating`, not yet billed).

    `DELETE /v1/files/<file id>` must be 400 with a message starting `Cannot delete file <file id>. The file
    is referenced by 1 batch(es) in non-terminal state: <batch id>: validating`, no `DELETE` reaches the
    provider, and the file row stays. Deleting the input early would make the batch impossible to bill.
    """
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        file_id: Final = _managed_file(rig, key, unique("referenced"))
        batch_id: Final = _managed_batch(rig, key, file_id)
        rig.wire.drain()
        response: Final = rig.proxy.request("DELETE", f"/v1/files/{file_id}", key=key)
        assert response.status_code == 400, response.text
        message: Final = str(error_of(response)["message"])
        assert (
            f"Cannot delete file {file_id}. The file is referenced by 1 batch(es) in non-terminal state: "
            f"{batch_id}: validating"
        ) in message
        assert [request for request in rig.drained() if request.method == "DELETE"] == []


def test_deleting_an_unused_managed_file_deletes_the_provider_file_and_the_row(rig: Rig) -> None:
    """A key uploads a managed file that no batch uses and deletes it.

    The answer must be 200 `{"id": <unified id>, "object": "file", "deleted": true}`, the provider must get
    `DELETE /v1/files/file-<stem>`, and the `LiteLLM_ManagedFileTable` row must be gone.
    """
    stem: Final = unique("unused")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        file_id: Final = _managed_file(rig, key, stem)
        rig.wire.drain()
        response: Final = rig.proxy.request("DELETE", f"/v1/files/{file_id}", key=key)
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["id"], body["object"], body["deleted"]) == (file_id, "file", True)
        assert ("DELETE", f"/v1/files/file-{stem}") in [(r.method, path_of(r)) for r in rig.drained()]
        assert (
            rig.proxy.rows(
                'SELECT unified_file_id FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s', (file_id,)
            )
            == []
        )


def test_model_routed_upload_rewrites_only_rows_naming_the_alias(rig: Rig) -> None:
    """A batch file has a row naming the proxy alias and a row naming `gpt-4.1-nano`; it is uploaded with the
    `model` form field set to the alias.

    The provider must receive the alias row with `body.model == "gpt-4o-mini"` (the deployment's provider
    model) written as compact JSON, and the other row byte for byte as uploaded, including its spacing (#51843).
    """
    alias_row: Final = batch_line(OPENAI_MODEL, custom_id=unique("alias-row"))
    other_line: Final = (
        b'{"custom_id": "other-row", "method": "POST", "url": "/v1/chat/completions", '
        b'"body": {"model": "gpt-4.1-nano", "messages": [{"role": "user", "content": "keep"}]}}\n'
    )
    content: Final = jsonl(alias_row) + other_line
    rig.wire.drain()
    response: Final = rig.upload(f"{unique('mixed')}.jsonl", content, model=OPENAI_MODEL)
    assert response.status_code == 200, response.text
    (upload,) = _uploads(rig)
    sent_lines: Final = multipart_field(upload, "file").split(b"\n")
    expected_alias: Final = {
        **alias_row,
        "body": {**JSON_OBJECT.validate_python(alias_row["body"]), "model": "gpt-4o-mini"},
    }
    assert sent_lines[0] == json.dumps(expected_alias, separators=(",", ":"), ensure_ascii=False).encode()
    assert sent_lines[1] + b"\n" == other_line


def test_model_routed_upload_without_alias_rows_is_sent_unchanged_under_its_own_name(rig: Rig) -> None:
    """Every row names `gpt-4.1-nano`, not the alias; the file is uploaded with `model` set to the alias.

    The provider must receive exactly the uploaded bytes under the caller's filename, not a rewritten
    `modified_file.jsonl`.
    """
    content: Final = jsonl(
        batch_line("gpt-4.1-nano", custom_id="nano-1"), batch_line("gpt-4.1-nano", custom_id="nano-2")
    )
    filename: Final = f"{unique('no-alias')}.jsonl"
    rig.wire.drain()
    response: Final = rig.upload(filename, content, model=OPENAI_MODEL)
    assert response.status_code == 200, response.text
    (upload,) = _uploads(rig)
    assert multipart_field(upload, "file") == content
    assert multipart_filename(upload) == filename


def test_managed_upload_reuploads_the_rows_under_the_deployment_model_as_modified_file(rig: Rig) -> None:
    """A key uploads a batch file whose rows name the proxy alias, with `target_model_names` set to it.

    The provider must receive `modified_file.jsonl` whose rows no longer name the alias and keep their
    `custom_id`s in order (`litellm/litellm_core_utils/batch_utils.py` `replace_model_in_jsonl`).
    UNVERIFIED: whether the new model is `gpt-4o-mini` or `openai/gpt-4o-mini`; both are accepted here.
    """
    first: Final = unique("managed-row-1")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        content: Final = jsonl(batch_line(OPENAI_MODEL, custom_id=first), batch_line(OPENAI_MODEL, custom_id="row-2"))
        rig.wire.drain()
        rig.uploaded_id("aliased.jsonl", content, key=key, target_model_names=OPENAI_MODEL)
        (upload,) = _uploads(rig)
        assert multipart_filename(upload) == "modified_file.jsonl"
        rows: Final = [json.loads(line) for line in multipart_field(upload, "file").split(b"\n") if line.strip()]
        assert [row["custom_id"] for row in rows] == [first, "row-2"]
        assert {row["body"]["model"] for row in rows} <= {"gpt-4o-mini", "openai/gpt-4o-mini"}, rows


def test_a_user_data_upload_reaches_the_provider_with_its_purpose_and_bytes(rig: Rig) -> None:
    """A virtual key uploads a text file with `purpose=user_data` and `custom_llm_provider=openai`.

    The provider must receive `purpose: user_data` and the exact bytes, and the caller gets a file object
    whose `purpose` is `user_data`. user_data files are not validated as batch JSONL.
    """
    content: Final = b"plain notes, not jsonl\n"
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.wire.drain()
        response: Final = rig.upload(
            f"{unique('notes')}.txt",
            content,
            purpose="user_data",
            key=key,
            content_type="text/plain",
            custom_llm_provider="openai",
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["purpose"] == "user_data"
        (upload,) = _uploads(rig)
        assert multipart_field(upload, "purpose") == b"user_data"
        assert multipart_field(upload, "file") == content
