import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from pydantic import SecretStr, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value, encrypt_value
from litellm.proxy.control_plane.settings import DataplaneSpec
from litellm.proxy.control_plane.store.models import ENTITY, Entity

_TEXT: Final = TypeAdapter(str)
_BYTES: Final = TypeAdapter(bytes)
_SECRET_FIELDS: Final = (("oidc_exchange", "client_secret"), ("service_token", "token"))


def _transform_field(name: str, value: object, transform: Callable[[str], str]) -> object:
    section, secret = _secret_field(name)
    if name != section or value is None:
        return value
    fields: Final = ENTITY.validate_python(value)
    return {**fields, secret: transform(_TEXT.validate_python(fields[secret]))}


def _secret_field(name: str) -> tuple[str, str]:
    return next((pair for pair in _SECRET_FIELDS if pair[0] == name), ("", ""))


@dataclass(frozen=True, slots=True)
class SecretCodec:
    key: SecretStr = field(repr=False)

    def encrypt(self, value: str) -> str:
        encrypted: Final = _BYTES.validate_python(encrypt_value(value, self.key.get_secret_value()))
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    def decrypt(self, value: str) -> str:
        return decrypt_value(base64.urlsafe_b64decode(value), self.key.get_secret_value())

    def encode(self, spec: DataplaneSpec) -> Entity:
        encoded: Final = ENTITY.validate_json(spec.model_dump_json())
        oidc: Final = spec.oidc_exchange
        service: Final = spec.service_token
        return {
            **encoded,
            **(
                {}
                if oidc is None
                else {
                    "oidc_exchange": {
                        **ENTITY.validate_python(encoded["oidc_exchange"]),
                        "client_secret": self.encrypt(oidc.client_secret.get_secret_value()),
                    }
                }
            ),
            **({} if service is None else {"service_token": {"token": self.encrypt(service.token.get_secret_value())}}),
        }

    def decode(self, entity: Entity) -> DataplaneSpec:
        return DataplaneSpec.model_validate(
            {name: _transform_field(name, value, self.decrypt) for name, value in entity.items()}
        )
