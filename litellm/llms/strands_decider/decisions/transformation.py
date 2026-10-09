import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, SignsRequestsWithAWS, sign_aws_json_post
from litellm.types.decisions import DecisionsIRRequest, DecisionsIRResponse
from litellm.types.llms.bedrock import AwsAuthParams

AGENTCORE_SESSION_HEADER: Final = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
_RUNTIME_ARN: Final = re.compile(
    r"\Aarn:aws(?:-[a-z]+)*:bedrock-agentcore:(?P<region>[a-z0-9-]+):[0-9]{12}:runtime/[A-Za-z0-9_-]+\Z"
)
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


class _RuntimeError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str = ""


class _RuntimeReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: _RuntimeError | None = None
    answers: object = None


def _runtime_error(payload: object) -> _RuntimeError | None:
    reply: Final = _RuntimeReply.model_validate(payload)
    return reply.error if reply.answers is None else None


def _with_session_id(headers: Mapping[str, str], runtime: AgentCoreRuntime) -> Mapping[str, str]:
    if any(name.lower() == AGENTCORE_SESSION_HEADER.lower() for name in headers):
        return headers
    return {**headers, AGENTCORE_SESSION_HEADER: runtime.default_session_id}


class StrandsDeciderDecisionsConfig(BaseDecisionsConfig, SignsRequestsWithAWS):
    api_key_env = ("STRANDS_DECIDER_API_KEY",)
    api_base_env = ("STRANDS_DECIDER_API_BASE",)
    api_key_required = False

    def __init__(self, aws: BaseAWSLLM | None = None) -> None:
        self._aws: Final = aws if aws is not None else BaseAWSLLM()

    def get_complete_url(self, api_base: str, model: str) -> str:
        runtime: Final = agentcore_runtime(api_base)
        return super().get_complete_url(api_base, model) if runtime is None else runtime.invocations_url

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
        session_headers: Final = _with_session_id(headers, runtime)
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

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        runtime_error: Final = _runtime_error(payload)
        if runtime_error is not None:
            raise BaseLLMException(
                status_code=_RUNTIME_ERROR_STATUS.get(runtime_error.code, 500),
                message=f"Strands Decider runtime error '{runtime_error.code}': {runtime_error.message}",
            )
        return super().parse_response(payload, request)
