import base64
import json
import struct
from collections.abc import Mapping
from typing import Final
from urllib.parse import unquote

from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

ACCESS_KEY: Final = "AKIAINTEGRATION000004"
SECRET_KEY: Final = "synthetic-secret-key-for-testing"
REGION: Final = "us-east-1"
COMPONENT: Final = "integration-embed-component"
SERVED_MODEL: Final = "integration-org/served-embedding-model"
VOYAGE_SERVED_MODEL: Final = "voyage-served-model"
ENDPOINT_PREFIX: Final = "integration-embed-"
COMPONENT_HEADER: Final = "x-amzn-sagemaker-inference-component"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
EXPECTED_SHAPE: Final = "Expected an OpenAI embeddings object with a 'data' list of {'embedding': [...]} rows"
NON_STRING_MODEL_ID: Final = "model_id must be the inference component name as a string"
IC_REQUIRED: Final = (
    "Inference Component Name header is required for endpoints to which you plan to deploy inference components. "
    "Please include Inference Component Name header or consider using SageMaker models."
)
IC_NOT_ALLOWED: Final = (
    "Inference Component Name header is not allowed for endpoints to which you dont plan to deploy inference "
    "components. Please remove the Inference Component Name header and try again."
)
PASS_IN_HINT: Final = "pass in via `litellm.embedding(..., model_id={InferenceComponentName})`"
REMOVE_HINT: Final = "remove `model_id` from this deployment, the endpoint has no inference components"


def endpoint(scenario: str) -> str:
    return f"{ENDPOINT_PREFIX}{scenario}"


def openai_route(scenario: str) -> str:
    return f"sagemaker/openai/{endpoint(scenario)}"


def plain_route(scenario: str) -> str:
    return f"sagemaker/{endpoint(scenario)}"


def invocations_target(name: str) -> str:
    return f"/endpoints/{name}/invocations"


def vector(index: int) -> list[float]:
    return [0.25, -0.5, 1.0 + index]


def inputs(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    carried: Final = next((body[key] for key in ("input", "inputs", "texts") if key in body), [])
    return tuple(carried) if isinstance(carried, list) else (carried,)


def embedding_value(index: int, encoding_format: JsonValue) -> JsonValue:
    floats: Final = vector(index)
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(floats)}f", *floats)).decode()
    return floats


def openai_rows(count: int, encoding_format: JsonValue, indexed: bool) -> list[JsonValue]:
    return [
        {
            "object": "embedding",
            **({"index": index} if indexed else {}),
            "embedding": embedding_value(index, encoding_format),
        }
        for index in range(count)
    ]


def _openai_usage(scenario: str) -> dict[str, JsonValue]:
    if scenario == "nousage":
        return {}
    if scenario == "promptonly":
        return {"usage": {"prompt_tokens": 5}}
    return {"usage": {"prompt_tokens": 7, "total_tokens": 7}}


def openai_payload(body: Mapping[str, JsonValue], scenario: str = "openai") -> dict[str, JsonValue]:
    return {
        "object": "list",
        "data": openai_rows(len(inputs(body)), body.get("encoding_format"), indexed=scenario != "noindex"),
        **({} if scenario == "nomodel" else {"model": SERVED_MODEL}),
        **_openai_usage(scenario),
    }


def json_reply(payload: JsonValue) -> Reply:
    return Reply(body=json.dumps(payload).encode())


def aws_error(status: int, code: str, message: str, **fields: JsonValue) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"__type": code, "Message": message, **fields}).encode(),
        headers={"x-amzn-ErrorType": code},
    )


def endpoint_name_of(request: Request) -> str:
    parts: Final = request.target.split("/")
    return unquote(parts[2]) if len(parts) > 2 else ""


def scenario_of(name: str) -> str:
    if not name.startswith(ENDPOINT_PREFIX):
        return ""
    return name.removeprefix(ENDPOINT_PREFIX).split("-")[0]


def respond(request: Request) -> Reply:
    name: Final = endpoint_name_of(request)
    body: Final = JSON_OBJECT.validate_json(request.body)
    match scenario_of(name):
        case "openai" | "nousage" | "nomodel" | "noindex" | "promptonly":
            return json_reply(openai_payload(body, scenario_of(name)))
        case "hf":
            return json_reply([vector(index) for index in range(len(inputs(body)))])
        case "icrequired":
            if COMPONENT_HEADER not in request.headers:
                return aws_error(400, "ValidationError", IC_REQUIRED)
            return json_reply(openai_payload(body))
        case "noic":
            if COMPONENT_HEADER in request.headers:
                return aws_error(400, "ValidationError", IC_NOT_ALLOWED)
            return json_reply([vector(index) for index in range(len(inputs(body)))])
        case "hfdict":
            return json_reply({"embedding": [vector(index) for index in range(len(inputs(body)))]})
        case "voyage":
            return json_reply(
                {
                    "object": "list",
                    "data": openai_rows(len(inputs(body)), None, indexed=True),
                    "model": VOYAGE_SERVED_MODEL,
                    "usage": {"total_tokens": 9},
                }
            )
        case "cohere":
            return json_reply(
                {
                    "id": "integration-cohere-response",
                    "texts": list(inputs(body)),
                    "embeddings": [vector(index) for index in range(len(inputs(body)))],
                    "meta": {"api_version": {"version": "1"}, "billed_units": {"input_tokens": 6}},
                    "response_type": "embeddings_floats",
                }
            )
        case "datadict":
            return json_reply({"object": "list", "data": {"embedding": vector(0)}})
        case "hfshape":
            return json_reply([vector(0)])
        case "notjson":
            return Reply(body=b"<html>upstream gateway error</html>", content_type="text/html")
        case "err400":
            return aws_error(
                400, "ValidationError", "1 validation error detected: Value at 'body' failed to satisfy constraint"
            )
        case "err424":
            return aws_error(
                424,
                "ModelError",
                'Received client error (400) from model with message "{\\"error\\":\\"unexpected field inputs\\"}".',
                OriginalStatusCode=400,
                OriginalMessage='{"error":"unexpected field inputs"}',
                LogStreamArn="arn:aws:logs:us-east-1:000000000000:log-group:/aws/sagemaker/Endpoints/integration",
            )
        case "err429":
            return aws_error(429, "ModelNotReadyException", "Model is not ready for inference yet")
        case "err503":
            return aws_error(503, "ServiceUnavailable", "The endpoint is scaling, try again later")
        case "err500":
            return aws_error(500, "InternalFailure", "An internal failure occurred")
        case "drop":
            return Reply(drop_connection=True)
        case _:
            return aws_error(400, "ValidationError", f"Endpoint {name} of account 000000000000 not found.")


def litellm_params(model: str, **parameters: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "aws_access_key_id": ACCESS_KEY,
        "aws_secret_access_key": SECRET_KEY,
        "aws_region_name": REGION,
        **parameters,
    }


def deployment(
    name: str, model: str, *, model_info: Mapping[str, JsonValue] | None = None, **parameters: JsonValue
) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": litellm_params(model, **parameters),
        **({} if model_info is None else {"model_info": dict(model_info)}),
    }


def openai_deployment(
    name: str, scenario: str, *, model_info: Mapping[str, JsonValue] | None = None, **parameters: JsonValue
) -> dict[str, JsonValue]:
    defaults: Final[dict[str, JsonValue]] = {"model_id": COMPONENT, "extra_body": {"model": SERVED_MODEL}}
    return deployment(name, openai_route(scenario), model_info=model_info, **{**defaults, **parameters})
