from collections.abc import Mapping, Sequence
from types import MappingProxyType

from pydantic import Field, field_validator, model_validator

from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.management_endpoints.auto_router_endpoints import DEFAULT_ROUTING_TEST_ROUTER_NAME


class RequestComplexityRouterConfig(ComplexityRouterConfig):
    """The part of a complexity-router config a request can carry.

    `plugins` holds live RoutingPlugin objects, which no JSON body can express and which have no
    OpenAPI schema, so it is closed off here rather than left as an arbitrary-type field.
    """

    plugins: None = Field(default=None, description="Not settable over HTTP; routing plugins are runtime objects")
    classifier_plugin: None = Field(  # pyright: ignore[reportIncompatibleVariableOverride]  # narrowing to None is the point: runtime objects are not settable over HTTP
        default=None, description="Not settable over HTTP; the classifier plugin is a runtime object"
    )


class AutoRouterRoutingTestRequest(LiteLLMBaseModel):
    """A single request to classify against a complexity-router config that need not be saved yet.

    Carries the same fields the serving path carries, so a dry run classifies what a real turn
    would classify. `messages`, `system` and `tools` are forwarded to the routing hook untranslated,
    which is why they are typed loosely: the hook reads whatever dialect the surface produced, and
    validating them against one surface's schema would reject the others.
    """

    prompt: str | None = Field(
        default=None,
        description="A single ask to route, as an end user would send it. Mutually exclusive with messages",
    )
    messages: Sequence[Mapping[str, object]] | None = Field(
        default=None,
        description="The full message list to route, exactly as the serving path would receive it. Mutually exclusive with prompt",
    )
    system: str | Sequence[Mapping[str, object]] | None = Field(
        default=None,
        description="The top-level system prompt an Anthropic /v1/messages body carries beside its messages",
    )
    tools: Sequence[Mapping[str, object]] | None = Field(
        default=None,
        description="The tool definitions the request advertises, which decide whether the plan-mode floor applies",
    )
    complexity_router_config: RequestComplexityRouterConfig = Field(
        description="The complexity router config to route against, in the shape /model/new accepts",
    )
    saved_model_id: str | None = Field(
        default=None,
        min_length=1,
        description="Test this saved deployment's server-side configuration instead of the supplied config and default model",
    )
    default_model: str | None = Field(
        default=None,
        description="Model to route to when no tier resolves, i.e. complexity_router_default_model",
    )
    router_name: str = Field(
        default=DEFAULT_ROUTING_TEST_ROUTER_NAME,
        description="Name reported as the router in the routing decision. Display only",
    )
    team_id: str | None = Field(
        default=None,
        description="Team the router is being created for. Required for a team admin, who may only test their own team's routers",
    )

    @field_validator("messages")
    @classmethod
    def _reject_messages_no_surface_accepts(
        cls, value: Sequence[Mapping[str, object]] | None
    ) -> Sequence[Mapping[str, object]] | None:
        """Reject what every supported surface rejects, and nothing beyond it.

        A real request carrying a message with no string role, or with content that is neither text
        nor a block list, is a 400 on the serving path, so answering it here with a routed tier
        would promise a decision the request never gets. Only the two keys the dialects agree on
        are constrained: anything else in a message stays untranslated and unread.
        """
        if value is None:
            return value
        for index, message in enumerate(value):
            if not isinstance(role := message.get("role"), str) or not role.strip():
                raise ValueError(f"messages[{index}] needs a non-empty string role")
            if (content := message.get("content")) is not None and not isinstance(content, str | list):
                raise ValueError(f"messages[{index}] content must be a string, a list of blocks, or null")
        return value

    @model_validator(mode="after")
    def _resolve_request_carrier(self) -> "AutoRouterRoutingTestRequest":
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if self.messages is not None and not self.messages:
            raise ValueError("messages must not be empty")
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        if self.messages is not None:
            return self
        return self.model_copy(update={"messages": [{"role": "user", "content": self.prompt}]})

    def wire_body(self) -> Mapping[str, object]:
        """The request kwargs a serving-path request would carry for this body.

        Every value is handed out by identity rather than copied, so the messages the routing hook
        classifies and the messages its raw-body plan-mode scan reads are one value, as they are on
        the serving path.
        """
        return MappingProxyType(
            {
                key: value
                for key, value in (("messages", self.messages), ("system", self.system), ("tools", self.tools))
                if value is not None
            }
        )
