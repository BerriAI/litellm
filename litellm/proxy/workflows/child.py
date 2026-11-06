from typing import Annotated, Final

from openai.types.chat import ChatCompletionMessageParam
from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.llms.custom_httpx.asgi_handler import get_async_asgi_client
from litellm.models.workflow import WorkflowExecution
from litellm.models.workflow_bundle import ChatCompletionNode, WorkflowFailure
from litellm.proxy._types import ProxyLifespanState
from litellm.proxy.workflows.auth import WorkflowChildGrant, workflow_child_grant
from litellm.proxy.workflows.runner import ChildResult
from litellm.proxy.workflows.validation import validate_value
from litellm.types.llms.base import LiteLLMBaseModel


class _Message(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None = None
    refusal: str | None = None


class _Choice(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    finish_reason: str
    message: _Message


class _Usage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0


class _Completion(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    choices: tuple[_Choice, ...]
    usage: _Usage = _Usage()


_MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])
_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COST: Final = TypeAdapter[float | None](Annotated[float | None, Field(ge=0, allow_inf_nan=False)])


class GatewayChildCaller:
    def __init__(self, app: ASGIApp, state: ProxyLifespanState) -> None:
        self._app: Final = app
        self._state: Final = state

    async def dispatch(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._app({**scope, "state": {**self._state}}, receive, send)

    async def call(
        self, execution: WorkflowExecution, node: ChatCompletionNode, messages: JsonValue, remaining: float
    ) -> ChildResult | WorkflowFailure:
        try:
            validated_messages: Final = _MESSAGES.validate_python(messages)
        except ValidationError:
            return WorkflowFailure(
                code="invalid_messages",
                message="Resolved messages must be standard Chat Completions messages",
                param=node.id,
            )
        original_request: Final = _OBJECT.validate_json(execution.request_json)
        payload: Final = {
            **node.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude={"id", "type", "messages", "retry", "timeout_seconds"},
            ),
            "messages": _MESSAGES.dump_python(validated_messages, mode="json"),
            "stream": False,
            "n": 1,
            "timeout": remaining,
            **({"user": original_request["user"]} if "user" in original_request else {}),
            "metadata": {
                "workflow_run_id": execution.run_id,
                "workflow_node_id": node.id,
                "workflow_snapshot_id": execution.snapshot_id,
            },
        }
        token: Final = workflow_child_grant.set(
            WorkflowChildGrant(
                principal=execution.principal,
                team_id=execution.team_id,
                snapshot_id=execution.snapshot_id,
                workflow_model=f"workflow/{execution.name}",
                node_id=node.id,
                model=node.model,
                run_id=execution.run_id,
            )
        )
        try:
            with get_async_asgi_client(app=self.dispatch) as client:
                response: Final = await client.post(
                    "http://litellm.internal/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer sk-workflow-internal", "x-litellm-session-id": execution.run_id},
                    timeout=remaining,
                )
            if not response.is_success:
                return WorkflowFailure(
                    code="child_call_failed",
                    message=f"Gateway child returned HTTP {response.status_code}",
                    param=node.id,
                    status_code=response.status_code,
                )
            try:
                completion: Final = _Completion.model_validate_json(response.content)
                response_cost: Final = _COST.validate_python(response.headers.get("x-litellm-response-cost"))
            except ValidationError:
                return WorkflowFailure(
                    code="invalid_completion",
                    message="Gateway child returned an invalid completion",
                    param=node.id,
                    status_code=502,
                )
            if len(completion.choices) != 1:
                return WorkflowFailure(
                    code="invalid_completion",
                    message="Workflow nodes require one completion choice",
                    param=node.id,
                    status_code=502,
                )
            choice: Final = completion.choices[0]
            if choice.finish_reason != "stop" or choice.message.refusal is not None or choice.message.content is None:
                return WorkflowFailure(
                    code="invalid_completion",
                    message="Completion refused, truncated, or produced no text result",
                    param=node.id,
                    status_code=502,
                )
            if node.response_format is None:
                return ChildResult(
                    output=choice.message.content,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    request_id=completion.id,
                    response_cost=response_cost,
                )
            try:
                output: Final = _JSON.validate_json(choice.message.content)
            except ValidationError:
                return WorkflowFailure(
                    code="invalid_json", message="Completion content is not valid JSON", param=node.id, status_code=502
                )
            failure: Final = validate_value(output, node.response_format.json_schema.schema_, node.id)
            return (
                failure
                if failure is not None
                else ChildResult(
                    output=output,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    request_id=completion.id,
                    response_cost=response_cost,
                )
            )
        finally:
            workflow_child_grant.reset(token)
