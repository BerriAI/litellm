from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import SecretStr

from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.kafka import KafkaInitParams

KafkaConfigValue: TypeAlias = str | int | bool
SecretReader: TypeAlias = Callable[[str], str | None]

_ENV_FIELDS: Final = MappingProxyType(
    {
        "bootstrap_servers": "KAFKA_BOOTSTRAP_SERVERS",
        "topic": "KAFKA_TOPIC",
        "security_protocol": "KAFKA_SECURITY_PROTOCOL",
        "sasl_mechanism": "KAFKA_SASL_MECHANISM",
        "sasl_username": "KAFKA_SASL_USERNAME",
        "sasl_password": "KAFKA_SASL_PASSWORD",
        "ssl_ca_location": "KAFKA_SSL_CA_LOCATION",
        "ssl_endpoint_identification_algorithm": "KAFKA_SSL_ENDPOINT_IDENTIFICATION_ALGORITHM",
        "ssl_certificate_location": "KAFKA_SSL_CERTIFICATE_LOCATION",
        "ssl_key_location": "KAFKA_SSL_KEY_LOCATION",
        "ssl_key_password": "KAFKA_SSL_KEY_PASSWORD",
    }
)


def _resolve(value: object, read_secret: SecretReader) -> object:
    raw: Final = value.get_secret_value() if isinstance(value, SecretStr) else value
    if isinstance(raw, str) and raw.startswith("os.environ/"):
        resolved: Final = read_secret(raw)
        if resolved is None:
            raise ValueError("A Kafka configuration environment reference is unset")
        return resolved
    return value


def resolve_params(
    configured: KafkaInitParams | Mapping[str, object] | None,
    read_secret: SecretReader = get_secret_str,
) -> KafkaInitParams:
    explicit: Final[Mapping[str, object] | None] = (
        configured.model_dump(exclude_unset=True) if isinstance(configured, KafkaInitParams) else configured
    )
    environment: Final = {
        name: value for name, variable in _ENV_FIELDS.items() if (value := read_secret(variable)) is not None
    }
    values: Final = {name: _resolve(value, read_secret) for name, value in (explicit or {}).items()}
    return KafkaInitParams.model_validate({**environment, **values})


def producer_config(params: KafkaInitParams) -> Mapping[str, KafkaConfigValue]:
    optional: Final[Mapping[str, str | None]] = MappingProxyType(
        {
            "sasl.mechanism": params.sasl_mechanism if params.security_protocol.startswith("SASL_") else None,
            "sasl.username": params.sasl_username if params.security_protocol.startswith("SASL_") else None,
            "sasl.password": params.sasl_password.get_secret_value() if params.sasl_password else None,
            "ssl.ca.location": params.ssl_ca_location,
            "ssl.certificate.location": params.ssl_certificate_location,
            "ssl.key.location": params.ssl_key_location,
            "ssl.key.password": params.ssl_key_password.get_secret_value() if params.ssl_key_password else None,
        }
    )
    return MappingProxyType(
        {
            "bootstrap.servers": params.bootstrap_servers,
            "client.id": params.client_id,
            "security.protocol": params.security_protocol,
            "enable.ssl.certificate.verification": True,
            "ssl.endpoint.identification.algorithm": params.ssl_endpoint_identification_algorithm,
            "compression.type": params.compression_type,
            "linger.ms": params.linger_ms,
            "batch.size": params.batch_size,
            "queue.buffering.max.messages": params.queue_max_messages,
            "queue.buffering.max.kbytes": params.queue_max_kbytes,
            "message.max.bytes": params.message_max_bytes,
            "delivery.timeout.ms": params.delivery_timeout_ms,
            "request.timeout.ms": params.request_timeout_ms,
            "retry.backoff.ms": params.retry_backoff_ms,
            "retry.backoff.max.ms": params.retry_backoff_max_ms,
            "enable.idempotence": params.enable_idempotence,
            "acks": "all",
            "log_level": 0,
            **{name: value for name, value in optional.items() if value is not None},
        }
    )
