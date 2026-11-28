from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import SecretStr, ValidationError

from litellm.integrations.kafka.config import producer_config, resolve_params
from litellm.types.integrations.kafka import KafkaInitParams


def test_explicit_settings_override_environment_and_resolve_secret_references() -> None:
    environment: Final = MappingProxyType(
        {
            "KAFKA_BOOTSTRAP_SERVERS": "environment-broker:9092",
            "KAFKA_TOPIC": "environment-topic",
            "KAFKA_SECURITY_PROTOCOL": "SASL_SSL",
            "KAFKA_SASL_USERNAME": "environment-user",
            "KAFKA_SASL_PASSWORD": "environment-password",
            "os.environ/TENANT_PASSWORD": "selected-password",
        }
    )
    params: Final = resolve_params(
        {
            "bootstrap_servers": "configured-broker:9093",
            "topic": "configured-topic",
            "sasl_password": "os.environ/TENANT_PASSWORD",
        },
        environment.get,
    )

    assert params.bootstrap_servers == "configured-broker:9093"
    assert params.topic == "configured-topic"
    assert params.sasl_username == "environment-user"
    assert params.sasl_password is not None
    assert params.sasl_password.get_secret_value() == "selected-password"
    assert "selected-password" not in repr(params)


def test_unset_environment_references_fail_without_using_the_reference_as_a_password() -> None:
    with pytest.raises(ValueError, match="environment reference is unset"):
        resolve_params({"topic": "events", "sasl_password": "os.environ/MISSING_PASSWORD"}, lambda _: None)


@pytest.mark.parametrize("mechanism", ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"])
def test_sasl_credentials_and_ca_are_forwarded_with_hostname_verification(mechanism: str) -> None:
    params: Final = KafkaInitParams.model_validate(
        {
            "bootstrap_servers": "broker:9093",
            "topic": "events",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": mechanism,
            "sasl_username": "producer-user",
            "sasl_password": "producer-password",
            "ssl_ca_location": "/certificates/ca.pem",
        }
    )
    config: Final = producer_config(params)

    assert config["sasl.mechanism"] == mechanism
    assert config["sasl.username"] == params.sasl_username
    assert config["sasl.password"] == "producer-password"
    assert config["ssl.ca.location"] == params.ssl_ca_location
    assert config["enable.ssl.certificate.verification"] is True
    assert config["ssl.endpoint.identification.algorithm"] == "https"


def test_mutual_tls_forwards_the_client_certificate_and_encrypted_key() -> None:
    params: Final = KafkaInitParams(
        bootstrap_servers="broker:9093",
        topic="events",
        security_protocol="SSL",
        ssl_certificate_location="/certificates/client.pem",
        ssl_key_location="/certificates/client.key",
        ssl_key_password=SecretStr("encrypted-key-password"),
    )
    config: Final = producer_config(params)

    assert config["ssl.certificate.location"] == params.ssl_certificate_location
    assert config["ssl.key.location"] == params.ssl_key_location
    assert params.ssl_key_password is not None
    assert config["ssl.key.password"] == params.ssl_key_password.get_secret_value()
    assert config["enable.ssl.certificate.verification"] is True
    assert config["ssl.endpoint.identification.algorithm"] == "https"


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        (None, "none"),
        ({"ssl_endpoint_identification_algorithm": "https"}, "https"),
        ({"ssl_endpoint_identification_algorithm": "os.environ/TENANT_HOSTNAME_POLICY"}, "none"),
    ),
)
def test_hostname_policy_preserves_ca_verification(configured: Mapping[str, object] | None, expected: str) -> None:
    environment: Final = MappingProxyType(
        {
            "KAFKA_BOOTSTRAP_SERVERS": "broker:9093",
            "KAFKA_TOPIC": "events",
            "KAFKA_SSL_CA_LOCATION": "/certificates/ca.pem",
            "KAFKA_SSL_ENDPOINT_IDENTIFICATION_ALGORITHM": "none",
            "os.environ/TENANT_HOSTNAME_POLICY": "none",
        }
    )
    params: Final = resolve_params(configured, environment.get)
    config: Final = producer_config(params)

    assert config["ssl.endpoint.identification.algorithm"] == expected
    assert config["enable.ssl.certificate.verification"] is True
    assert config["ssl.ca.location"] == params.ssl_ca_location


@pytest.mark.parametrize("protocol", [None, "PLAINTEXT"])
def test_transport_defaults_to_verified_tls_and_plaintext_requires_opt_in(protocol: str | None) -> None:
    params: Final = resolve_params(
        {
            "bootstrap_servers": "broker:9092",
            "topic": "events",
            **({"security_protocol": protocol} if protocol else {}),
        },
        lambda _: None,
    )
    config: Final = producer_config(params)

    assert config["security.protocol"] == (protocol or "SSL")
    assert config["enable.ssl.certificate.verification"] is True
    assert config["ssl.endpoint.identification.algorithm"] == "https"


def test_plaintext_configuration_does_not_send_a_sasl_mechanism() -> None:
    config: Final = producer_config(
        KafkaInitParams(bootstrap_servers="broker:9092", topic="events", security_protocol="PLAINTEXT")
    )

    assert not any(name.startswith("sasl.") for name in config)


def test_queue_limits_and_delivery_deadline_reach_the_client() -> None:
    params: Final = KafkaInitParams(
        bootstrap_servers="broker:9092",
        topic="events",
        queue_max_messages=12,
        queue_max_kbytes=1024,
        delivery_timeout_ms=4000,
        request_timeout_ms=1000,
        retry_backoff_ms=200,
        retry_backoff_max_ms=1500,
    )
    config: Final = producer_config(params)

    assert config["queue.buffering.max.messages"] == params.queue_max_messages
    assert config["queue.buffering.max.kbytes"] == params.queue_max_kbytes
    assert config["delivery.timeout.ms"] == params.delivery_timeout_ms
    assert config["retry.backoff.ms"] == params.retry_backoff_ms
    assert config["retry.backoff.max.ms"] == params.retry_backoff_max_ms
    assert config["acks"] == "all"
    assert config["enable.idempotence"] is params.enable_idempotence


@pytest.mark.parametrize(
    "changes",
    (
        {"topic": "../events"},
        {"topic": "."},
        {"bootstrap_servers": ""},
        {"queue_max_messages": 0},
        {"queue_max_kbytes": 0},
        {"callback_queue_max_messages": 0},
        {"security_protocol": "SASL_SSL"},
        {"sasl_password": "unused-password"},
        {"ssl_certificate_location": "/certificates/client.pem"},
        {"ssl_endpoint_identification_algorithm": "disabled"},
        {"delivery_timeout_ms": 100, "request_timeout_ms": 100},
        {"retry_backoff_ms": 500, "retry_backoff_max_ms": 100},
        {"flush_timeout": float("inf")},
        {"unsupported_option": "unrecognized"},
    ),
)
def test_invalid_configuration_is_rejected_before_export(changes: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        KafkaInitParams.model_validate({"bootstrap_servers": "broker:9092", "topic": "events", **changes})


def test_validation_errors_and_representations_do_not_disclose_passwords() -> None:
    password: Final = "do-not-log-this-password"
    with pytest.raises(ValidationError) as error:
        KafkaInitParams.model_validate(
            {
                "bootstrap_servers": "broker:9093",
                "topic": "invalid/topic",
                "security_protocol": "SASL_SSL",
                "sasl_username": "user",
                "sasl_password": password,
            }
        )

    assert password not in str(error.value)
    assert password not in repr(SecretStr(password))
