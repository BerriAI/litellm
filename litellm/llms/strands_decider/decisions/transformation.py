import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final
from urllib.parse import quote, unquote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, sign_aws_json_post
from litellm.types.decisions import DecisionsIRRequest, DecisionsIRResponse
from litellm.types.llms.bedrock import AwsAuthParams

AGENTCORE_SESSION_HEADER: Final = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
_RESERVED_HEADER_PREFIXES: Final = ("x-amzn-bedrock-agentcore-runtime-", "x-amz-")
_RUNTIME_ARN: Final = re.compile(
    r"\Aarn:aws(?:-[a-z]+)*:bedrock-agentcore:(?P<region>[a-z0-9-]+):[0-9]{12}:runtime/[A-Za-z0-9_-]+\Z"
)
_INVOCATIONS_PATH: Final = re.compile(r"\A/runtimes/(?P<arn>[^/]+)/invocations\Z")
_RUNTIME_ERROR_STATUS: Final = MappingProxyType({"bad_request": 400, "invalid_request": 400, "loading": 503})


@dataclass(frozen=True, slots=True)
class AgentCoreRuntime:
    arn: str
    region: str

    @property
    def invocations_url(self) -> str:
        host: Final = f"bedrock-agentcore.{self.region}.{get_aws_dns_suffix(self.region)}"
        return f"https://{host}/runtimes/{quote(self.arn, safe='')}/invocations"

    @property
    def default_session_id(self) -> str:
        return f"litellm-decider-{hashlib.sha256(self.arn.encode()).hexdigest()}"


def agentcore_runtime(api_base: str) -> AgentCoreRuntime | None:
    match: Final = _RUNTIME_ARN.match(api_base)
    return None if match is None else AgentCoreRuntime(arn=api_base, region=match["region"])


def invoked_runtime(url: httpx.URL) -> AgentCoreRuntime | None:
    match: Final = _INVOCATIONS_PATH.match(url.raw_path.decode("ascii"))
    runtime: Final = None if match is None else agentcore_runtime(unquote(match["arn"]))
    return runtime if runtime is not None and runtime.invocations_url == str(url) else None


class _RuntimeError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str = ""


class _RuntimeReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: _RuntimeError | None = None
    answers: object = None


def _runtime_error(content: bytes) -> _RuntimeError | None:
    try:
        reply: Final = _RuntimeReply.model_validate_json(content)
    except ValidationError:
        return None
    return reply.error if reply.answers is None else None


class _SessionParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    agentcore_runtime_session_id: str | None = None


def _is_reserved_header(name: str) -> bool:
    lowered: Final = name.lower()
    return lowered == "host" or lowered.startswith(_RESERVED_HEADER_PREFIXES)


def _runtime_headers(
    headers: Mapping[str, str], runtime: AgentCoreRuntime, litellm_params: Mapping[str, object]
) -> Mapping[str, str]:
    dropped: Final = sorted(name for name in headers if _is_reserved_header(name))
    if dropped:
        verbose_logger.warning("Strands Decider: not forwarding reserved AgentCore header(s) %s", dropped)
    configured: Final = _SessionParams.model_validate(litellm_params).agentcore_runtime_session_id
    return {
        **{name: value for name, value in headers.items() if not _is_reserved_header(name)},
        AGENTCORE_SESSION_HEADER: runtime.default_session_id if configured is None else configured,
    }


class StrandsDeciderDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("STRANDS_DECIDER_API_KEY",)
    api_base_env = ("STRANDS_DECIDER_API_BASE",)
    api_key_required = False

    def __init__(self, aws: BaseAWSLLM | None = None) -> None:
        self._aws: Final = aws if aws is not None else BaseAWSLLM()

    def get_complete_url(self, api_base: str, model: str) -> str:
        runtime: Final = agentcore_runtime(api_base)
        return super().get_complete_url(api_base, model) if runtime is None else runtime.invocations_url

    def signs_with_aws(self, api_base: str) -> bool:
        return agentcore_runtime(api_base) is not None

    def sign_request(
        self,
        headers: Mapping[str, str],
        url: str,
        api_base: str,
        body: Mapping[str, object],
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[Mapping[str, str], bytes | None]:
        runtime: Final = agentcore_runtime(api_base)
        if runtime is None:
            return headers, None
        session_headers: Final = _runtime_headers(headers, runtime, litellm_params)
        if api_key is not None:
            return session_headers, None
        payload: Final = json.dumps(body)
        signed: Final = sign_aws_json_post(
            get_credentials=partial(
                self._aws.resolve_credentials, AwsAuthParams.model_validate(litellm_params), runtime.region
            ),
            service_name="bedrock-agentcore",
            aws_region_name=runtime.region,
            url=url,
            body=payload,
            headers=session_headers,
        )
        return dict(signed.headers.items()), payload.encode()

    def transform_decisions_response(
        self,
        model: str,
        custom_llm_provider: str,
        raw_response: httpx.Response,
        request: DecisionsIRRequest,
    ) -> DecisionsIRResponse:
        runtime_error: Final = (
            None if invoked_runtime(raw_response.request.url) is None else _runtime_error(raw_response.content)
        )
        if runtime_error is not None:
            raise BaseLLMException(
                status_code=_RUNTIME_ERROR_STATUS.get(runtime_error.code, 500),
                message=f"Strands Decider runtime error '{runtime_error.code}': {runtime_error.message}",
            )
        return super().transform_decisions_response(model, custom_llm_provider, raw_response, request)
