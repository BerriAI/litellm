from typing import Literal

from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class KafkaInitParams(StandardCustomLoggerInitParams):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)

    bootstrap_servers: str = Field(min_length=1)
    topic: str = Field(pattern=r"^[a-zA-Z0-9._-]{1,249}$")
    security_protocol: Literal["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"] = "SSL"
    sasl_mechanism: Literal["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"] = "PLAIN"
    sasl_username: str | None = None
    sasl_password: SecretStr | None = None
    ssl_ca_location: str | None = None
    ssl_endpoint_identification_algorithm: Literal["https", "none"] = "https"
    ssl_certificate_location: str | None = None
    ssl_key_location: str | None = None
    ssl_key_password: SecretStr | None = None
    client_id: str = "litellm"
    compression_type: Literal["none", "gzip", "snappy", "lz4", "zstd"] = "gzip"
    linger_ms: int = Field(default=20, ge=0, le=900_000)
    batch_size: int = Field(default=65_536, gt=0)
    queue_max_messages: int = Field(default=10_000, gt=0)
    callback_queue_max_messages: int = Field(default=128, gt=0)
    queue_max_kbytes: int = Field(default=65_536, gt=0)
    message_max_bytes: int = Field(default=1_000_000, ge=1_000)
    delivery_timeout_ms: int = Field(default=30_000, gt=0)
    request_timeout_ms: int = Field(default=10_000, gt=0)
    retry_backoff_ms: int = Field(default=100, ge=0)
    retry_backoff_max_ms: int = Field(default=1_000, ge=0)
    enable_idempotence: bool = True
    flush_timeout: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def connection_settings(self) -> Self:
        if self.topic in (".", ".."):
            raise ValueError("Kafka topic cannot be . or ..")
        if self.security_protocol.startswith("SASL_") and (not self.sasl_username or not self.sasl_password):
            raise ValueError("Kafka SASL requires sasl_username and sasl_password")
        if not self.security_protocol.startswith("SASL_") and (self.sasl_username or self.sasl_password):
            raise ValueError("Kafka SASL credentials require SASL_SSL or SASL_PLAINTEXT")
        if bool(self.ssl_certificate_location) != bool(self.ssl_key_location):
            raise ValueError("Kafka client TLS requires both ssl_certificate_location and ssl_key_location")
        if self.delivery_timeout_ms < self.request_timeout_ms + self.linger_ms:
            raise ValueError("delivery_timeout_ms must cover request_timeout_ms plus linger_ms")
        if self.retry_backoff_max_ms < self.retry_backoff_ms:
            raise ValueError("retry_backoff_max_ms must cover retry_backoff_ms")
        return self
