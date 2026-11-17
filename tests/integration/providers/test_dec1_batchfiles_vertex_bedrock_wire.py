"""Contract: batch files and batch jobs on Vertex AI (GCS + batchPredictionJobs) and Bedrock (S3 + model
invocation jobs), seen through the OpenAI `/v1/files` and `/v1/batches` routes.

Vertex: a `purpose=batch` upload to a `vertex_ai/` deployment is not sent to a files API. The proxy converts
each OpenAI row to a Vertex request row (`{"request": {<gemini body>, "labels": {"litellm_custom_id": ...}}}`)
and writes the result to the deployment's `gcs_bucket_name` with
`POST {api_base}/upload/storage/v1/b/<bucket>/o?uploadType=media&name=litellm-vertex-files/publishers/google/models/<model>/<uuid>`;
the file id is the `gs://` URI. Creating a batch posts a batch prediction job whose output prefix is the input
file's directory; retrieve is a `GET` on the job, `JOB_STATE_*` maps to an OpenAI status, and the output file
is `<gcsOutputDirectory>/predictions.jsonl`. Cancel posts `<job>:cancel` with `{}` and then re-reads the job.

Bedrock: the upload is written with a signed `PUT {s3_endpoint_url}/<bucket>/litellm-bedrock-files-<model>-<uuid>.jsonl`
holding `{recordId, modelInput}` rows. Creating a batch posts `CreateModelInvocationJob`
(`POST {aws_bedrock_control_plane_endpoint}/model-invocation-job`) with `modelId`, a `litellm-batch-` job
name, the input S3 URI, an output prefix `s3://<bucket>/litellm-batch-outputs/<jobName>/`, `roleArn` and
`timeoutDurationInHours: 24`. Bedrock statuses map onto OpenAI ones; cancel is `StopModelInvocationJob`
followed by a status read, and a stop that conflicts with a job already in a terminal state answers that state.

Read from `litellm/llms/vertex_ai/batches/{handler,transformation}.py`, `litellm/llms/vertex_ai/files/transformation.py:1132`,
`litellm/llms/bedrock/batches/{handler,transformation}.py`, `litellm/llms/bedrock/files/transformation.py:505`
and `litellm/litellm_core_utils/cloud_storage_security.py`. Assumed: `aws_bedrock_control_plane_endpoint` and
`s3_endpoint_url` may point at a plain-HTTP loopback peer (boto3 and the SigV4 signer accept http), and the
Vertex rows helper's token endpoint on the shared upstream answers `scripted-token`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final
from urllib.parse import unquote

import pytest
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_batchfiles_support import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    BEDROCK_JOB_ARN_PREFIX,
    BEDROCK_MODEL_ID,
    BEDROCK_ROLE_ARN,
    GCS_BUCKET,
    JSON_OBJECT,
    S3_BUCKET,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    Script,
    bedrock_job,
    body_of,
    chat_row,
    decoded,
    json_of,
    jsonl,
    path_of,
    reply,
    sent,
    unique,
    vertex_job,
)
from integration.providers.dec1_vertex4_support import owned_config
from pydantic import JsonValue

_VERTEX_BACKEND: Final = "gemini-2.5-flash"
_JOBS_PATH: Final = f"/v1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/batchPredictionJobs"
_GCS_UPLOAD_PATH: Final = f"/upload/storage/v1/b/{GCS_BUCKET}/o"
_GCS_PREFIX: Final = f"litellm-vertex-files/publishers/google/models/{_VERTEX_BACKEND}/"


@pytest.fixture(scope="module")
def allowlisted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """An owned proxy whose `user_url_allowed_hosts` names 127.0.0.1.

    Vertex batch retrieve reads the job through `safe_get` (litellm/llms/vertex_ai/batches/handler.py:410-418),
    which refuses a loopback `api_base` with a 500 "URL targets a blocked address" unless the host is allowlisted
    (`_is_host_allowlisted`, litellm/litellm_core_utils/url_utils.py:307-321). The shared proxy has no allowlist,
    so the retrieve contracts run here; create, upload and cancel do not go through `safe_get` and stay on the
    shared proxy.
    """
    directory: Final = tmp_path_factory.mktemp("dec1-batchfiles-vertex")
    config: Final = owned_config(directory, user_url_allowed_hosts=["127.0.0.1"])
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config, workers=1) as proxy:
        yield proxy


def _gcs_stored(request: Request) -> Reply:
    name: Final = request.target.split("name=", 1)[1]
    plain: Final = unquote(name)
    return reply(
        {
            "kind": "storage#object",
            "id": f"{GCS_BUCKET}/{plain}/1",
            "name": plain,
            "bucket": GCS_BUCKET,
            "size": str(len(request.body)),
            "timeCreated": "2026-11-17T10:00:00.000Z",
            "contentType": "application/octet-stream",
        }
    )


def _vertex_deployment(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"vertex_ai/{_VERTEX_BACKEND}",
        api_key=None,
        api_base=wire.url,
        vertex_project=VERTEX_PROJECT,
        vertex_location=VERTEX_LOCATION,
        vertex_credentials=service_account_json(VERTEX_PROJECT, gateway.upstream_url),
        gcs_bucket_name=GCS_BUCKET,
    )


def _vertex_upload(gateway: Gateway, model: str, *rows: dict[str, JsonValue]) -> str:
    response: Final = gateway.request_multipart(
        "/v1/files", {"purpose": "batch", "model": model}, {"file": ("rows.jsonl", jsonl(*rows), "application/jsonl")}
    )
    assert response.status_code == 200, response.text
    identity: Final = json_of(response)["id"]
    assert isinstance(identity, str)
    return identity


def _gcs_uploads(wire: Wire) -> list[Request]:
    return sent(wire.drain(), "POST", _GCS_UPLOAD_PATH)


def _vertex_script() -> Script:
    script: Final = Script()
    script.on("POST", _GCS_UPLOAD_PATH, _gcs_stored)
    return script


def test_vertex_batch_upload_is_written_to_the_deployment_bucket_under_the_managed_model_prefix(
    gateway: Gateway,
) -> None:
    """A vertex deployment with `gcs_bucket_name` exists; the caller uploads a one-row batch file naming it.

    Exactly one GCS media upload must reach `/upload/storage/v1/b/<bucket>/o` with `uploadType=media` and an
    object name under `litellm-vertex-files/publishers/google/models/gemini-2.5-flash/`; the returned file id
    decodes to that `gs://` URI. Nothing goes to a `/v1/files` route.
    """
    script: Final = _vertex_script()
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire)
        file_id: Final = _vertex_upload(gateway, model, chat_row("r1", model))
        requests: Final = wire.drain()
        uploads: Final = sent(requests, "POST", _GCS_UPLOAD_PATH)
        assert len(uploads) == 1, [request.target for request in requests]
        assert "uploadType=media" in uploads[0].target
        name: Final = JSON_OBJECT.validate_json(_gcs_stored(uploads[0]).body)["name"]
        assert isinstance(name, str) and name.startswith(_GCS_PREFIX), name
        assert f"gs://{GCS_BUCKET}/{name}" in decoded(file_id.removeprefix("file-"))
        assert [request for request in requests if path_of(request).endswith("/files")] == []


def test_vertex_upload_rows_become_gemini_requests_labelled_with_their_custom_id(gateway: Gateway) -> None:
    """The caller uploads two OpenAI chat rows (`custom_id` r1 and r2, `max_tokens: 16`).

    The stored object must have two JSON lines; each has `request.contents[0]` with `role: "user"` and the
    row's text as `parts[0].text`, `request.generationConfig.max_output_tokens == 16`, and
    `request.labels.litellm_custom_id` equal to the row's `custom_id`, in input order. The label is how output
    rows are matched back to OpenAI `custom_id`s. The generation config keeps the snake_case key that
    `transform_request_body` writes (litellm/llms/vertex_ai/files/transformation.py:894-900); Vertex parses
    batch rows as proto JSON, which accepts the original field names as well as lowerCamelCase.
    """
    script: Final = _vertex_script()
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire)
        _vertex_upload(gateway, model, chat_row("r1", model, "first"), chat_row("r2", model, "second"))
        (upload,) = _gcs_uploads(wire)
        lines: Final = [json.loads(line) for line in upload.body.decode().split("\n") if line.strip()]
        assert [line["request"]["labels"]["litellm_custom_id"] for line in lines] == ["r1", "r2"]
        assert [line["request"]["contents"][0]["parts"][0]["text"] for line in lines] == ["first", "second"]
        assert {line["request"]["contents"][0]["role"] for line in lines} == {"user"}
        assert {line["request"]["generationConfig"]["max_output_tokens"] for line in lines} == {16}


def _vertex_batch(gateway: Gateway, script: Script, wire: Wire, model: str, job_id: str) -> tuple[str, str]:
    """Upload, then create a batch whose scripted job id is `job_id`; returns (encoded batch id, gs:// input)."""
    file_id: Final = _vertex_upload(gateway, model, chat_row("r1", model))
    (upload,) = _gcs_uploads(wire)
    name: Final = JSON_OBJECT.validate_json(_gcs_stored(upload).body)["name"]
    input_uri: Final = f"gs://{GCS_BUCKET}/{name}"
    script.on("POST", _JOBS_PATH, reply(vertex_job(job_id, "JOB_STATE_PENDING", input_uri=input_uri)))
    created: Final = gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert created.status_code == 200, created.text
    identity: Final = json_of(created)["id"]
    assert isinstance(identity, str)
    return identity, input_uri


def test_vertex_batch_create_posts_a_prediction_job_reading_the_uploaded_object(gateway: Gateway) -> None:
    """The caller uploads a batch file and creates a batch from the returned id, naming no model.

    The upstream must get one `POST /v1/projects/<p>/locations/<l>/batchPredictionJobs` with
    `inputConfig.gcsSource.uris == [<gs:// input>]`, `instancesFormat: jsonl`,
    `outputConfig.gcsDestination.outputUriPrefix` equal to the input's directory and
    `model == "publishers/google/models/gemini-2.5-flash"`; the caller gets `status: validating`.
    """
    script: Final = _vertex_script()
    job_id: Final = unique("7412")
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire)
        _batch_id, input_uri = _vertex_batch(gateway, script, wire, model, job_id)
        (post,) = sent(wire.drain(), "POST", _JOBS_PATH)
        body: Final = body_of(post)
        assert body["inputConfig"] == {"gcsSource": {"uris": [input_uri]}, "instancesFormat": "jsonl"}
        assert body["outputConfig"] == {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": input_uri.rsplit("/", 1)[0]},
        }
        assert body["model"] == f"publishers/google/models/{_VERTEX_BACKEND}"


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("JOB_STATE_QUEUED", "validating"),
        ("JOB_STATE_RUNNING", "in_progress"),
        ("JOB_STATE_PAUSED", "in_progress"),
        ("JOB_STATE_UPDATING", "in_progress"),
        ("JOB_STATE_CANCELLING", "cancelling"),
        ("JOB_STATE_CANCELLED", "cancelled"),
        ("JOB_STATE_FAILED", "failed"),
        ("JOB_STATE_EXPIRED", "expired"),
        ("JOB_STATE_UNSPECIFIED", "failed"),
    ],
)
def test_vertex_retrieve_maps_each_job_state(allowlisted: Gateway, state: str, status: str) -> None:
    """A Vertex batch exists and the job is in the given `JOB_STATE_*`.

    `GET /v1/batches/<id>` must reach `GET .../batchPredictionJobs/<job id>` and answer the mapped status.
    `JOB_STATE_UNSPECIFIED` maps to `failed`, not `validating`, unlike Gemini's unspecified state.
    """
    gateway: Final = allowlisted
    script: Final = _vertex_script()
    job_id: Final = unique("7413")
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire)
        batch_id, input_uri = _vertex_batch(gateway, script, wire, model, job_id)
        script.on("GET", f"{_JOBS_PATH}/{job_id}", reply(vertex_job(job_id, state, input_uri=input_uri)))
        wire.drain()
        retrieved: Final = gateway.request("GET", f"/v1/batches/{batch_id}")
        assert retrieved.status_code == 200, retrieved.text
        assert json_of(retrieved)["status"] == status
        assert sent(wire.drain(), "GET", f"{_JOBS_PATH}/{job_id}"), "the job was never read"


def test_vertex_output_file_is_predictions_jsonl_under_the_reported_output_directory(allowlisted: Gateway) -> None:
    """The running job reports `outputInfo.gcsOutputDirectory: gs://<bucket>/out/prediction-1/`.

    The caller's batch must carry an `output_file_id` that decodes to
    `gs://<bucket>/out/prediction-1/predictions.jsonl` (trailing slash dropped, file name appended); before
    Vertex reports `outputInfo` the field is null.
    """
    gateway: Final = allowlisted
    script: Final = _vertex_script()
    job_id: Final = unique("7414")
    directory: Final = f"gs://{GCS_BUCKET}/out/prediction-1/"
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire)
        batch_id, input_uri = _vertex_batch(gateway, script, wire, model, job_id)
        script.on("GET", f"{_JOBS_PATH}/{job_id}", reply(vertex_job(job_id, "JOB_STATE_RUNNING", input_uri=input_uri)))
        assert json_of(gateway.request("GET", f"/v1/batches/{batch_id}"))["output_file_id"] is None
        script.on(
            "GET",
            f"{_JOBS_PATH}/{job_id}",
            reply(vertex_job(job_id, "JOB_STATE_RUNNING", input_uri=input_uri, output_directory=directory)),
        )
        output_file_id: Final = json_of(gateway.request("GET", f"/v1/batches/{batch_id}"))["output_file_id"]
        assert isinstance(output_file_id, str)
        assert f"gs://{GCS_BUCKET}/out/prediction-1/predictions.jsonl" in decoded(output_file_id.removeprefix("file-"))


def test_vertex_cancel_posts_cancel_with_an_empty_body_and_then_reads_the_job(gateway: Gateway) -> None:
    """The caller cancels a Vertex batch.

    The upstream must get `POST .../batchPredictionJobs/<id>:cancel` with body `{}` followed by
    `GET .../batchPredictionJobs/<id>`; the caller's answer is the re-read job (`JOB_STATE_CANCELLING`
    becomes `cancelling`), not the empty cancel response.
    """
    script: Final = _vertex_script()
    job_id: Final = unique("7415")
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire)
        batch_id, input_uri = _vertex_batch(gateway, script, wire, model, job_id)
        script.on("POST", f"{_JOBS_PATH}/{job_id}:cancel", reply({}))
        script.on(
            "GET", f"{_JOBS_PATH}/{job_id}", reply(vertex_job(job_id, "JOB_STATE_CANCELLING", input_uri=input_uri))
        )
        wire.drain()
        cancelled: Final = gateway.request("POST", f"/v1/batches/{batch_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert json_of(cancelled)["status"] == "cancelling"
        requests: Final = tuple(request for request in wire.drain() if "/_oauth/" not in request.target)
        trail: Final = [(request.method, path_of(request)) for request in requests]
        assert trail[-2:] == [("POST", f"{_JOBS_PATH}/{job_id}:cancel"), ("GET", f"{_JOBS_PATH}/{job_id}")], trail
        (cancel,) = sent(requests, "POST", f"{_JOBS_PATH}/{job_id}:cancel")
        assert body_of(cancel) == {}


def _bedrock_deployment(scenario: Scenario, wire: Wire, **overrides: JsonValue) -> str:
    return scenario.model(
        model=f"bedrock/{BEDROCK_MODEL_ID}",
        api_key=None,
        api_base=None,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_region_name=AWS_REGION,
        s3_bucket_name=S3_BUCKET,
        s3_endpoint_url=wire.url,
        aws_bedrock_control_plane_endpoint=wire.url,
        **{"aws_batch_role_arn": BEDROCK_ROLE_ARN, **overrides},
    )


class _Bedrock:
    """A scripted S3 bucket and Bedrock control plane on one peer; PUTs to the bucket are kept by key."""

    def __init__(self) -> None:
        self.script: Final = Script()
        self.objects: dict[str, bytes] = {}
        self.job_arn: Final = BEDROCK_JOB_ARN_PREFIX + unique("litellm-batch-job")

    def __call__(self, request: Request) -> Reply:
        path: Final = path_of(request)
        if request.method == "PUT" and path.startswith(f"/{S3_BUCKET}/"):
            self.objects[path.removeprefix(f"/{S3_BUCKET}/")] = request.body
            return Reply(body=b"", headers={"ETag": '"dec1"'})
        if request.method == "POST" and path == "/model-invocation-job":
            return reply({"jobArn": self.job_arn})
        return self.script(request)

    def status(self, status: str) -> None:
        self.script.on(
            "GET",
            f"/model-invocation-job/{self.job_arn}",
            reply(
                bedrock_job(
                    self.job_arn,
                    status,
                    input_uri=f"s3://{S3_BUCKET}/input.jsonl",
                    output_uri=f"s3://{S3_BUCKET}/litellm-batch-outputs/out/",
                )
            ),
        )


def _bedrock_batch(gateway: Gateway, model: str) -> str:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model},
        {"file": ("in.jsonl", jsonl(chat_row("req-1", model, "ping")), "application/jsonl")},
    )
    assert uploaded.status_code == 200, uploaded.text
    created: Final = gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": json_of(uploaded)["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert created.status_code == 200, created.text
    identity: Final = json_of(created)["id"]
    assert isinstance(identity, str)
    return identity


def test_bedrock_batch_upload_is_one_s3_put_of_record_id_and_model_input_rows(gateway: Gateway) -> None:
    """A bedrock deployment with `s3_bucket_name` exists; a managed batch upload names it.

    Exactly one object must be PUT into the bucket, its key starting with `litellm-bedrock-files-` and ending
    in `.jsonl`, holding one JSON line whose `recordId` is the row's `custom_id` (`req-1`) and whose
    `modelInput` is the Anthropic Messages body (`messages[0].role == "user"`, `max_tokens == 16`).
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        uploaded: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("in.jsonl", jsonl(chat_row("req-1", model, "ping")), "application/jsonl")},
        )
        assert uploaded.status_code == 200, uploaded.text
        assert len(bedrock.objects) == 1, list(bedrock.objects)
        ((object_key, content),) = bedrock.objects.items()
        assert object_key.startswith("litellm-bedrock-files-") and object_key.endswith(".jsonl"), object_key
        (record,) = [json.loads(line) for line in content.decode().split("\n") if line.strip()]
        assert record["recordId"] == "req-1"
        assert record["modelInput"]["messages"][0]["role"] == "user"
        assert record["modelInput"]["max_tokens"] == 16


def test_bedrock_batch_create_posts_a_model_invocation_job_with_role_and_output_prefix(gateway: Gateway) -> None:
    """The caller creates a batch from a managed Bedrock batch file.

    The control plane must get one `POST /model-invocation-job` whose JSON has `modelId` equal to the
    deployment's model id (no `bedrock/` prefix), `roleArn` equal to `aws_batch_role_arn`, a `jobName`
    starting with `litellm-batch-`, `inputDataConfig.s3InputDataConfig.s3Uri` equal to the uploaded object,
    `outputDataConfig.s3OutputDataConfig.s3Uri == s3://<bucket>/litellm-batch-outputs/<jobName>/` and
    `timeoutDurationInHours: 24` (from `completion_window: 24h`). The request is SigV4 signed for `bedrock`.
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        _bedrock_batch(gateway, model)
        (post,) = sent(wire.drain(), "POST", "/model-invocation-job")
        body: Final = body_of(post)
        job_name: Final = body["jobName"]
        assert isinstance(job_name, str) and job_name.startswith("litellm-batch-"), job_name
        (object_key,) = bedrock.objects
        assert (body["modelId"], body["roleArn"], body["timeoutDurationInHours"]) == (
            BEDROCK_MODEL_ID,
            BEDROCK_ROLE_ARN,
            24,
        )
        assert body["inputDataConfig"] == {"s3InputDataConfig": {"s3Uri": f"s3://{S3_BUCKET}/{object_key}"}}
        assert body["outputDataConfig"] == {
            "s3OutputDataConfig": {"s3Uri": f"s3://{S3_BUCKET}/litellm-batch-outputs/{job_name}/"}
        }
        assert "/bedrock/aws4_request" in post.headers.get("authorization", "")


@pytest.mark.parametrize(
    ("bedrock_status", "status"),
    [
        ("Stopping", "cancelling"),
        ("Stopped", "cancelled"),
        ("Failed", "failed"),
        ("Expired", "expired"),
        ("SomethingNew", "in_progress"),
    ],
)
def test_bedrock_retrieve_maps_terminal_and_unknown_statuses(
    gateway: Gateway, bedrock_status: str, status: str
) -> None:
    """A Bedrock batch exists and `GetModelInvocationJob` reports the given status.

    The caller must see the mapped OpenAI status; a status Bedrock adds later falls back to `in_progress`
    instead of failing the retrieve. The proxy retrieves a model-invocation-job ARN through
    `handle_model_invocation_job_status` (litellm/batches/main.py:645-653), whose fallback is `in_progress`
    (litellm/llms/bedrock/batches/handler.py:374), not the `validating` fallback of
    `BedrockBatchesConfig.transform_retrieve_batch_response`. (`Submitted`, `Validating`, `Scheduled` and `InProgress` are pinned in
    `test_bedrock_batch_retrieve_status_wire.py`.)
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        batch_id: Final = _bedrock_batch(gateway, model)
        bedrock.status(bedrock_status)
        retrieved: Final = gateway.request("GET", f"/v1/batches/{batch_id}")
        assert retrieved.status_code == 200, retrieved.text
        assert json_of(retrieved)["status"] == status


def test_bedrock_retrieve_exposes_job_details_as_string_metadata(gateway: Gateway) -> None:
    """`GetModelInvocationJob` reports `jobArn`, `jobName`, `modelId`, the input URI and the output prefix.

    The caller's batch `metadata` carries them as strings under the names `handle_model_invocation_job_status`
    gives them (litellm/llms/bedrock/batches/handler.py:399-407): `model_arn`, `job_arn`, `job_name`,
    `input_s3_uri`, `output_s3_uri`, plus an empty `failure_message`, because the OpenAI batch schema only allows
    string metadata values. `roleArn` and `timeoutDurationInHours` are not surfaced on this path.
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        batch_id: Final = _bedrock_batch(gateway, model)
        bedrock.status("InProgress")
        metadata: Final = json_of(gateway.request("GET", f"/v1/batches/{batch_id}"))["metadata"]
        assert isinstance(metadata, dict), metadata
        assert (
            metadata["model_arn"],
            metadata["job_arn"],
            metadata["job_name"],
            metadata["input_s3_uri"],
            metadata["output_s3_uri"],
            metadata["failure_message"],
        ) == (
            BEDROCK_MODEL_ID,
            bedrock.job_arn,
            bedrock.job_arn.rsplit("/", 1)[-1],
            f"s3://{S3_BUCKET}/input.jsonl",
            f"s3://{S3_BUCKET}/litellm-batch-outputs/out/",
            "",
        ), metadata
        assert all(isinstance(value, str) for value in metadata.values()), metadata


def test_bedrock_cancel_stops_the_job_and_answers_the_status_read_after(gateway: Gateway) -> None:
    """The caller cancels a running Bedrock batch.

    The control plane must get `POST /model-invocation-job/<job arn>/stop` and then a status read; the caller
    gets the re-read job, here `Stopping`, as `cancelling`.
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        batch_id: Final = _bedrock_batch(gateway, model)
        bedrock.script.on("POST", f"/model-invocation-job/{bedrock.job_arn}/stop", reply({}))
        bedrock.status("Stopping")
        wire.drain()
        cancelled: Final = gateway.request("POST", f"/v1/batches/{batch_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert json_of(cancelled)["status"] == "cancelling"
        trail: Final = [(request.method, path_of(request)) for request in wire.drain()]
        assert ("POST", f"/model-invocation-job/{bedrock.job_arn}/stop") in trail, trail
        assert trail[-1] == ("GET", f"/model-invocation-job/{bedrock.job_arn}"), trail


def test_bedrock_cancel_of_an_already_failed_job_answers_its_state_instead_of_an_error(gateway: Gateway) -> None:
    """The job already failed, so `StopModelInvocationJob` answers 400 `ConflictException`.

    The proxy must read the job, see a terminal status (`failed`, in `_CANCEL_IDEMPOTENT_STATUSES`) and answer
    200 with `status: failed` instead of surfacing the conflict, so cancelling twice or late is safe.
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        batch_id: Final = _bedrock_batch(gateway, model)
        bedrock.script.on(
            "POST",
            f"/model-invocation-job/{bedrock.job_arn}/stop",
            Reply(
                status=400,
                body=json.dumps({"message": "job is not running"}).encode(),
                headers={"x-amzn-ErrorType": "ConflictException"},
            ),
        )
        bedrock.status("Failed")
        cancelled: Final = gateway.request("POST", f"/v1/batches/{batch_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert json_of(cancelled)["status"] == "failed"


def test_bedrock_batch_create_without_a_role_arn_fails_before_the_control_plane(gateway: Gateway) -> None:
    """The bedrock deployment has no `aws_batch_role_arn` and the proxy has no `AWS_BATCH_ROLE_ARN`.

    The create must fail with a message containing `AWS IAM role ARN is required for Bedrock batch jobs` and
    no `POST /model-invocation-job` may be sent. UNVERIFIED: the status code (the transformation raises
    `ValueError`), and that the shared proxy has no `AWS_BATCH_ROLE_ARN` in its environment.
    """
    bedrock: Final = _Bedrock()
    with wire_server(bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire, aws_batch_role_arn=None)
        uploaded: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("in.jsonl", jsonl(chat_row("req-1", model)), "application/jsonl")},
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = gateway.request(
            "POST",
            "/v1/batches",
            {"input_file_id": json_of(uploaded)["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        )
        assert created.status_code >= 400, created.text
        assert "AWS IAM role ARN is required for Bedrock batch jobs" in created.text
        assert sent(wire.drain(), "POST", "/model-invocation-job") == []
