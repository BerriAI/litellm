from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    SecretStr,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    WrapValidator,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from litellm.types.llms.base_model import LiteLLMBaseModel
from litellm.types.mcp import MCPAuth, MCPTokenEndpointAuthMethod

_FIELDS: Final = TypeAdapter(Mapping[str, object])


@contextmanager
def _redact_profile_errors() -> Generator[None, None, None]:
    try:
        yield
    except ValidationError as exc:
        # Both the rejected input and the location of an unknown field can contain credentials.
        errors: Final = [
            InitErrorDetails(
                type=PydanticCustomError(
                    "invalid_auth_profile",
                    "{reason}",
                    {"reason": "Unknown auth type" if error["type"] == "union_tag_invalid" else error["msg"]},
                ),
                loc=() if error["type"] in {"extra_forbidden", "invalid_key"} else error["loc"],
                input=None,
            )
            for error in exc.errors(include_input=False, include_context=False)
        ]
        raise ValidationError.from_exception_data("MCPAuthProfile", errors) from None


def _validate_profile(value: object, handler: Callable[[object], "_AuthProfile"]) -> "_AuthProfile":
    with _redact_profile_errors():
        return handler(value)


class _AuthProfile(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")

    if not TYPE_CHECKING:
        # Wrap the inherited descriptor so raw JSON errors are redacted with this Pydantic version's signature.
        model_validate_json = classmethod(_redact_profile_errors()(LiteLLMBaseModel.model_validate_json.__func__))

    @model_validator(mode="wrap")
    @classmethod
    def _redact_errors(cls, value: object, handler: ModelWrapValidatorHandler["_AuthProfile"]) -> "_AuthProfile":
        return _validate_profile(value, handler)

    @field_validator("id")
    @classmethod
    def _reject_reserved_id(cls, value: str) -> str:
        if value.lower() == "default":
            raise ValueError("the default credential has no profile id")
        return value


class StaticAuthProfile(_AuthProfile):
    auth_type: Literal[MCPAuth.api_key, MCPAuth.bearer_token, MCPAuth.basic, MCPAuth.token, MCPAuth.authorization]
    authentication_token: SecretStr | None = Field(
        default=None, exclude=True, repr=False, validation_alias=AliasChoices("authentication_token", "auth_value")
    )


class ClientCredentialsAuthProfile(_AuthProfile):
    auth_type: Literal[MCPAuth.oauth2]
    oauth2_flow: Literal["client_credentials"]
    token_url: str | None = None
    scopes: tuple[str, ...] | None = None
    token_endpoint_auth_method: MCPTokenEndpointAuthMethod | None = None
    client_id: SecretStr | None = Field(default=None, exclude=True, repr=False)
    client_secret: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @model_serializer(mode="wrap")
    def _omit_absent_settings(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202  # Retain output schema
        return {
            key: value
            for key, value in _FIELDS.validate_python(handler(self)).items()
            if value is not None or key in self.model_fields_set
        }


MCPAuthProfile: TypeAlias = Annotated[
    StaticAuthProfile | ClientCredentialsAuthProfile,
    Field(discriminator="auth_type"),
    WrapValidator(_validate_profile),
]


_PROFILE: Final[TypeAdapter[MCPAuthProfile]] = TypeAdapter(MCPAuthProfile)


def parse_auth_profile_json(data: str | bytes | bytearray) -> MCPAuthProfile:
    """Read credential-bearing JSON, including parser errors that precede model validation."""
    with _redact_profile_errors():
        return _PROFILE.validate_json(data)
