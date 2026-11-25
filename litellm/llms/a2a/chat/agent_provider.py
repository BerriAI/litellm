"""
Chat completions for registered agents whose agent type names an A2A provider
(``pydantic_ai_agents``, Bedrock AgentCore, watsonx Orchestrate), sent through that
provider's own client instead of the generic A2A request.
"""

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ..common_utils import A2AError, convert_messages_to_prompt
from .streaming_iterator import A2AModelResponseIterator
from .transformation import A2AConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# langflow's provider calls litellm.completion itself, so it keeps the generic A2A request
_PROVIDERS_THAT_CALL_COMPLETION: Final = frozenset({"langflow"})


def registered_agent_provider(agent_name: str) -> tuple[BaseA2AProviderConfig, Mapping[str, object]] | None:
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    except ImportError:
        return None
    agent: Final = global_agent_registry.get_agent_by_name(agent_name)
    agent_params: Final[Mapping[str, object]] = (agent.litellm_params if agent else None) or {}
    provider: Final = agent_params.get("custom_llm_provider")
    agent_type_model: Final = agent_params.get("model")
    if not isinstance(provider, str) or provider in _PROVIDERS_THAT_CALL_COMPLETION:
        return None
    config: Final = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider=provider,
        model=agent_type_model if isinstance(agent_type_model, str) else None,
    )
    return None if config is None else (config, agent_params)


def _message_params(messages: Sequence[AllMessageValues]) -> Mapping[str, object]:
    message: Final = {
        "kind": "message",
        "role": "user",
        "parts": [{"kind": "text", "text": convert_messages_to_prompt(list(messages))}],
        "messageId": str(uuid.uuid4()),
    }
    # fasta2a rejects a configuration without acceptedOutputModes
    return {"message": message, "configuration": {"acceptedOutputModes": ["text"], "blocking": True}}


class _AgentEventIterator(A2AModelResponseIterator):
    def __init__(
        self,
        lines: AsyncIterator[str],
        source: AsyncIterator[Mapping[str, object]],
    ) -> None:
        super().__init__(streaming_response=lines, sync_stream=False)  # pyright: ignore[reportUnknownMemberType]  # A2AModelResponseIterator.__init__ is untyped
        self._source: Final = source

    async def aclose(self) -> None:
        if isinstance(self._source, AsyncGenerator):
            await self._source.aclose()  # pyright: ignore[reportUnknownMemberType]  # the base class declares provider streams as AsyncIterator


def _raise_on_error_event(event: Mapping[str, object] | None) -> None:
    error: Final = None if event is None else event.get("error")
    if error is None:
        return
    message: Final = (as_str_mapping(error) or {}).get("message")
    raise A2AError(status_code=500, message=f"A2A error: {message if isinstance(message, str) else 'Unknown error'}")


async def _replayed(
    first: Mapping[str, object] | None, rest: AsyncIterator[Mapping[str, object]]
) -> AsyncIterator[str]:
    if first is not None:
        yield json.dumps(first)
    async for event in rest:
        yield json.dumps(event)


async def acompletion_through_agent_provider(
    provider: BaseA2AProviderConfig,
    agent_params: Mapping[str, object],
    model: str,
    messages: Sequence[AllMessageValues],
    api_base: str,
    api_key: str | None,
    headers: Mapping[str, object],
    stream: bool,
    fake_stream: bool,
    timeout: float | httpx.Timeout,
    model_response: ModelResponse,
    logging_obj: "LiteLLMLoggingObj",
) -> ModelResponse | CustomStreamWrapper:
    request_id: Final = str(uuid.uuid4())
    params: Final = _message_params(messages)
    agent_headers: Final = {
        **{name: value for name, value in headers.items() if isinstance(value, str)},
        **({} if api_key is None else {"Authorization": f"Bearer {api_key}"}),
    }
    logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # Logging.pre_call is untyped
        input=list(messages),
        api_key=None,
        additional_args={"complete_input_dict": params, "api_base": api_base},
    )
    if stream and not fake_stream:
        events: Final = aiter(
            provider.handle_streaming(  # pyright: ignore[reportUnknownMemberType]  # base class exposes **kwargs
                request_id=request_id,
                params=dict(params),
                api_base=api_base,
                litellm_params=agent_params,
                agent_extra_headers=agent_headers,
                timeout=timeout,
            )
        )
        first_event: Final = await anext(events, None)
        _raise_on_error_event(first_event)
        return CustomStreamWrapper(
            completion_stream=_AgentEventIterator(_replayed(first_event, events), source=events),
            model=model,
            custom_llm_provider="a2a",
            logging_obj=logging_obj,
        )
    reply: Final = await provider.handle_non_streaming(  # pyright: ignore[reportUnknownMemberType]  # base class exposes **kwargs
        request_id=request_id,
        params=dict(params),
        api_base=api_base,
        litellm_params=agent_params,
        agent_extra_headers=agent_headers,
        timeout=timeout,
    )
    answer: Final = A2AConfig().model_response_from_reply(
        reply=reply,
        model_response=model_response,
        model=model,
        messages=messages,
        status_code=500,
        headers={},
    )
    if not stream:
        return answer
    return CustomStreamWrapper(
        completion_stream=MockResponseIterator(model_response=answer),
        model=model,
        custom_llm_provider="a2a",
        logging_obj=logging_obj,
    )
