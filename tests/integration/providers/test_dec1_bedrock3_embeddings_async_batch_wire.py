"""Bedrock embeddings outside the plain `/invoke` call: async invoke (Nova 2 multimodal) and batch files (Titan v2).

`bedrock/async_invoke/<model>` on `/v1/embeddings` starts an asynchronous Bedrock job: the proxy must
`POST /async-invoke` with `{"modelId", "modelInput", "outputDataConfig": {"s3OutputDataConfig": {"s3Uri"}}}`,
where `modelInput` is the Nova multimodal body with `taskType: SEGMENTED_EMBEDDING`
(`litellm/llms/bedrock/embed/amazon_nova_transformation.py:86`), and return a placeholder embedding (an
empty vector, zero usage) because the vectors land in S3 later. Without `output_s3_uri` the proxy refuses
before calling Bedrock. The plain `bedrock/<nova embeddings id>` call uses `SINGLE_EMBEDDING` instead.

A managed batch file for a Titan Text Embeddings V2 deployment is converted row by row into Bedrock's
batch JSONL (`recordId` plus a `modelInput` that is the Titan v2 InvokeModel body) and PUT into the
deployment's S3 bucket (`BedrockFilesConfig._map_openai_embedding_to_bedrock_params`,
`litellm/llms/bedrock/files/transformation.py:1013`). Rows with several inputs, token id inputs or a
non Titan v2 model are refused, since Bedrock batch takes one `inputText` per record. All AWS hosts are
replaced by an owned peer (`api_base`, `s3_endpoint_url`, `aws_bedrock_control_plane_endpoint`).
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_bedrock3_support import unique_text
from integration.providers.dec1_bedrock_support import TITAN_EMBED_V2, bedrock, body_of, path_of
from pydantic import JsonValue

NOVA_EMBED: Final = "amazon.nova-2-multimodal-embeddings-v1:0"
S3_URI: Final = "s3://synthetic-dec1-bucket/async-out/"
INVOCATION_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/dec1job"
BUCKET: Final = "synthetic-dec1-batch-bucket"
BATCH_ACCESS_KEY: Final = "AKIADEC1BEDROCK3BT01"
PNG_BASE64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="


def _started(_: Request) -> Reply:
    return Reply(body=json.dumps({"invocationArn": INVOCATION_ARN}).encode())


# ------------------------------------------------------------------ Async invoke


def test_async_invoke_text_posts_a_segmented_nova_body_with_the_s3_output(gateway: Gateway) -> None:
    """`bedrock/async_invoke/amazon.nova-2-multimodal-embeddings-v1:0` with `output_s3_uri` embeds one text.

    Bedrock must receive one `POST /async-invoke` with `modelId` the bare Nova id, `outputDataConfig ==
    {"s3OutputDataConfig": {"s3Uri": <uri>}}`, and `modelInput` with `schemaVersion: nova-multimodal-embed-v1`,
    `taskType: SEGMENTED_EMBEDDING` and `segmentedEmbeddingParams` holding `embeddingPurpose: GENERIC_INDEX`,
    `embeddingDimension: 3072` and `text: {"value": <text>, "truncationMode": "END"}`.
    """
    text: Final = unique_text()
    with bedrock(gateway, _started, model=f"bedrock/async_invoke/{NOVA_EMBED}", output_s3_uri=S3_URI) as rig:
        response: Final = rig.embed(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == "/async-invoke"
        body: Final = body_of(sent)
        assert body["modelId"] == NOVA_EMBED
        assert body["outputDataConfig"] == {"s3OutputDataConfig": {"s3Uri": S3_URI}}
        model_input: Final = body["modelInput"]
        assert isinstance(model_input, dict)
        assert (model_input["schemaVersion"], model_input["taskType"]) == (
            "nova-multimodal-embed-v1",
            "SEGMENTED_EMBEDDING",
        )
        params: Final = model_input["segmentedEmbeddingParams"]
        assert isinstance(params, dict)
        assert (params["embeddingPurpose"], params["embeddingDimension"]) == ("GENERIC_INDEX", 3072)
        assert params["text"] == {"value": text, "truncationMode": "END"}


def test_async_invoke_dimensions_become_the_nova_embedding_dimension(gateway: Gateway) -> None:
    """`dimensions: 1024` on the async invoke embedding.

    It must reach Bedrock as `modelInput.segmentedEmbeddingParams.embeddingDimension == 1024` and no
    `dimensions` key may be left in those params.
    """
    with bedrock(gateway, _started, model=f"bedrock/async_invoke/{NOVA_EMBED}", output_s3_uri=S3_URI) as rig:
        response: Final = rig.embed(unique_text(), dimensions=1024)
        assert response.status_code == 200, response.text
        params: Final = body_of(rig.one_post())["modelInput"]["segmentedEmbeddingParams"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert params["embeddingDimension"] == 1024  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert "dimensions" not in params  # pyright: ignore[reportOperatorIssue]


def test_async_invoke_image_data_uri_becomes_an_image_source(gateway: Gateway) -> None:
    """The embedding input is `data:image/png;base64,<payload>`.

    Nova needs the modality spelled out, so `segmentedEmbeddingParams.image` must be
    `{"format": "png", "source": {"bytes": <payload>}}` and no `text` member may be sent.
    """
    with bedrock(gateway, _started, model=f"bedrock/async_invoke/{NOVA_EMBED}", output_s3_uri=S3_URI) as rig:
        response: Final = rig.embed(f"data:image/png;base64,{PNG_BASE64}")
        assert response.status_code == 200, response.text
        params: Final = body_of(rig.one_post())["modelInput"]["segmentedEmbeddingParams"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert params["image"] == {"format": "png", "source": {"bytes": PNG_BASE64}}  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert "text" not in params  # pyright: ignore[reportOperatorIssue]


def test_async_invoke_answers_a_placeholder_embedding_with_zero_usage(gateway: Gateway) -> None:
    """Bedrock answers `{"invocationArn": ...}` only.

    The caller must get HTTP 200 with one `data` entry whose `embedding` is `[]` and `usage.prompt_tokens: 0`,
    since the real vectors arrive in S3 (`_transform_async_invoke_response`, `amazon_nova_transformation.py:337`).
    """
    with bedrock(gateway, _started, model=f"bedrock/async_invoke/{NOVA_EMBED}", output_s3_uri=S3_URI) as rig:
        response: Final = rig.embed(unique_text())
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert [item["embedding"] for item in body["data"]] == [[]]
        assert body["usage"]["prompt_tokens"] == 0


def test_async_invoke_without_an_output_s3_uri_never_reaches_bedrock(gateway: Gateway) -> None:
    """The async invoke deployment has no `output_s3_uri`.

    `_wrap_async_invoke_request` raises `output_s3_uri is required for async invoke requests`, so the caller
    gets an error status naming `output_s3_uri` and Bedrock receives no POST.
    UNVERIFIED: the exact status (a 400 or a 500 from the unmapped ValueError).
    """
    with bedrock(gateway, _started, model=f"bedrock/async_invoke/{NOVA_EMBED}") as rig:
        response: Final = rig.embed(unique_text())
        assert response.status_code >= 400, response.text
        assert "output_s3_uri" in response.text
        assert rig.posts() == ()


def test_a_plain_nova_embedding_uses_single_embedding_on_invoke(gateway: Gateway) -> None:
    """`bedrock/amazon.nova-2-multimodal-embeddings-v1:0` (no `async_invoke/`).

    Bedrock must receive `POST /model/amazon.nova-2-multimodal-embeddings-v1:0/invoke` (decoded) with
    `taskType: SINGLE_EMBEDDING` and the params under `singleEmbeddingParams`. Only the request is pinned.
    """
    text: Final = unique_text()

    def respond(_: Request) -> Reply:
        return Reply(body=json.dumps({"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2]}]}).encode())

    with bedrock(gateway, respond, model=f"bedrock/{NOVA_EMBED}") as rig:
        rig.embed(text)
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{NOVA_EMBED}/invoke"
        body: Final = body_of(sent)
        assert body["taskType"] == "SINGLE_EMBEDDING"
        assert body["singleEmbeddingParams"]["text"] == {"value": text, "truncationMode": "END"}  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


# ------------------------------------------------------------------ Batch embeddings through managed files


class _Bucket:
    """A scripted S3 bucket and Bedrock control plane on one peer; object PUTs are kept by key."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def __call__(self, request: Request) -> Reply:
        path: Final = path_of(request)
        if request.method == "PUT" and path.startswith(f"/{BUCKET}/"):
            self.objects[path.removeprefix(f"/{BUCKET}/")] = request.body
            return Reply(body=b"", headers={"ETag": '"dec1"'})
        return Reply(body=b"{}")


def _upload(gateway: Gateway, bucket: _Bucket, model_id: str, *rows: dict[str, JsonValue]):  # noqa: ANN202
    """Register a Bedrock embedding deployment backed by `bucket` and upload `rows` as a batch file."""
    with wire_server(bucket) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{model_id}",
            api_key=None,
            api_base=None,
            aws_access_key_id=BATCH_ACCESS_KEY,
            aws_secret_access_key="synthetic-batch-secret-dec1",
            aws_region_name="us-east-1",
            s3_bucket_name=BUCKET,
            s3_endpoint_url=wire.url,
            aws_bedrock_control_plane_endpoint=wire.url,
            aws_batch_role_arn="arn:aws:iam::123456789012:role/dec1-bedrock3-batch",
        )
        lines: Final = "\n".join(json.dumps({**row, "body": {"model": model, **row["body"]}}) for row in rows)  # pyright: ignore[reportGeneralTypeIssues]
        return gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("embed.jsonl", lines.encode(), "application/jsonl")},
        )


def _embedding_row(custom_id: str, **body: JsonValue) -> dict[str, JsonValue]:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/embeddings", "body": dict(body)}


def _records(bucket: _Bucket) -> list[dict[str, JsonValue]]:
    assert len(bucket.objects) == 1, list(bucket.objects)
    (content,) = bucket.objects.values()
    return [json.loads(line) for line in content.decode().split("\n") if line.strip()]


def test_a_titan_v2_embedding_batch_row_becomes_an_input_text_model_input(gateway: Gateway) -> None:
    """A batch file with one `/v1/embeddings` row (`input: "hello batch"`, `dimensions: 256`) for Titan v2.

    The single S3 object must hold `{"recordId": "emb-1", "modelInput": {"inputText": "hello batch",
    "dimensions": 256}}`, the same body the synchronous Titan v2 path sends.
    """
    bucket: Final = _Bucket()
    response: Final = _upload(
        gateway, bucket, TITAN_EMBED_V2, _embedding_row("emb-1", input="hello batch", dimensions=256)
    )
    assert response.status_code == 200, response.text
    (record,) = _records(bucket)
    assert record == {"recordId": "emb-1", "modelInput": {"inputText": "hello batch", "dimensions": 256}}, record


def test_a_single_element_list_input_is_unwrapped_to_one_input_text(gateway: Gateway) -> None:
    """The row's `input` is `["only one"]`.

    `_coerce_embedding_input_to_string` unwraps a one item list, so `modelInput.inputText` must be `only one`.
    """
    bucket: Final = _Bucket()
    response: Final = _upload(gateway, bucket, TITAN_EMBED_V2, _embedding_row("emb-2", input=["only one"]))
    assert response.status_code == 200, response.text
    assert _records(bucket)[0]["modelInput"] == {"inputText": "only one"}


def test_a_float_encoding_row_asks_titan_for_float_embedding_types(gateway: Gateway) -> None:
    """The row sets `encoding_format: "float"`.

    It must map through the Titan v2 config to `modelInput.embeddingTypes == ["float"]`.
    """
    bucket: Final = _Bucket()
    response: Final = _upload(
        gateway, bucket, TITAN_EMBED_V2, _embedding_row("emb-3", input="x", encoding_format="float")
    )
    assert response.status_code == 200, response.text
    assert _records(bucket)[0]["modelInput"] == {"inputText": "x", "embeddingTypes": ["float"]}


def test_a_multi_input_embedding_row_is_refused_without_writing_to_s3(gateway: Gateway) -> None:
    """The row's `input` is `["a", "b"]`.

    Bedrock batch takes one input per record, so the upload must fail with a message containing
    `requires one input per JSONL record` and nothing may be PUT into the bucket.
    UNVERIFIED: the exact status (400 or 500).
    """
    bucket: Final = _Bucket()
    response: Final = _upload(gateway, bucket, TITAN_EMBED_V2, _embedding_row("emb-4", input=["a", "b"]))
    assert response.status_code >= 400, response.text
    assert "requires one input per JSONL" in response.text
    assert bucket.objects == {}


def test_a_token_id_embedding_row_is_refused_without_writing_to_s3(gateway: Gateway) -> None:
    """The row's `input` is token ids `[101, 102]`.

    Titan v2 batch has no token id input, so the upload must fail naming `pre-tokenized` and nothing may be
    PUT. UNVERIFIED: whether a list of ints first trips the multi-input check (both refuse).
    """
    bucket: Final = _Bucket()
    response: Final = _upload(gateway, bucket, TITAN_EMBED_V2, _embedding_row("emb-5", input=[101, 102]))
    assert response.status_code >= 400, response.text
    assert bucket.objects == {}


def test_a_non_titan_v2_embedding_batch_is_refused(gateway: Gateway) -> None:
    """A batch embedding row for `cohere.embed-english-v3`, which has no batch body mapping yet.

    The upload must fail with a message containing `currently supports only Amazon Titan Text Embeddings V2`
    and nothing may be PUT into the bucket.
    """
    bucket: Final = _Bucket()
    response: Final = _upload(gateway, bucket, "cohere.embed-english-v3", _embedding_row("emb-6", input="x"))
    assert response.status_code >= 400, response.text
    assert "currently supports only Amazon" in response.text
    assert bucket.objects == {}


def test_a_batch_upload_never_calls_the_runtime(gateway: Gateway) -> None:
    """A valid Titan v2 batch upload.

    Uploading only writes S3; no InvokeModel `POST /model/...` may reach the peer.
    """
    bucket: Final = _Bucket()
    calls: Final[list[Request]] = []

    def recording(request: Request) -> Reply:
        calls.append(request)
        return bucket(request)

    with wire_server(recording) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{TITAN_EMBED_V2}",
            api_key=None,
            api_base=None,
            aws_access_key_id=BATCH_ACCESS_KEY,
            aws_secret_access_key="synthetic-batch-secret-dec1",
            aws_region_name="us-east-1",
            s3_bucket_name=BUCKET,
            s3_endpoint_url=wire.url,
            aws_bedrock_control_plane_endpoint=wire.url,
            aws_batch_role_arn="arn:aws:iam::123456789012:role/dec1-bedrock3-batch",
        )
        row: Final = json.dumps({**_embedding_row("emb-7", input="x"), "body": {"model": model, "input": "x"}})
        response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("embed.jsonl", row.encode(), "application/jsonl")},
        )
        assert response.status_code == 200, response.text
    assert not [request for request in calls if path_of(request).startswith("/model/")], [
        (request.method, request.target) for request in calls
    ]
