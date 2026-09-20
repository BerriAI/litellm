import pytest

from litellm.callbacks_v1.builtin import openmeter
from litellm.callbacks_v1.builtin.port import CallRecord
from litellm.callbacks_v1.builtin.runtime import CallJoin
from tests.test_litellm.callbacks_v1.builtin.support import FIXED_NOW, golden_call

PORT = openmeter.OpenMeter(openmeter.Config(api_key="test-key", endpoint="http://openmeter.test/"))


def joined(terminal: str, **metadata: str) -> CallRecord:
    join = CallJoin()
    records = [join.accept(envelope) for envelope in golden_call(terminal, **metadata)]  # pyright: ignore[reportArgumentType]  # terminal is an EventName literal at every call site
    (record,) = [record for record in records if record is not None]
    return record


def test_a_succeeded_call_becomes_one_cloud_event() -> None:
    assert PORT.payload(joined("call.succeeded", user_api_key_user_id="user-1"), FIXED_NOW) == {
        "specversion": "1.0",
        "type": "litellm_tokens",
        "id": "response",
        "time": "2026-01-02T03:04:05+00:00",
        "subject": "user-1",
        "source": "litellm-proxy",
        "data": {"model": "model", "cost": None},
    }


def test_a_failed_call_is_not_metered() -> None:
    assert PORT.payload(joined("call.failed", user_api_key_user_id="user-1"), FIXED_NOW) is None


def test_a_call_without_a_subject_is_an_error_like_the_legacy_logger() -> None:
    with pytest.raises(ValueError, match="user is required"):
        PORT.payload(joined("call.succeeded"), FIXED_NOW)


def test_each_event_is_its_own_authorised_request() -> None:
    (delivery,) = PORT.deliveries(({"id": "a"},))  # pyright: ignore[reportArgumentType]  # framing does not depend on the event's fields

    assert delivery.url == "http://openmeter.test/api/v1/events"
    assert dict(delivery.headers) == {
        "Content-Type": "application/cloudevents+json",
        "Authorization": "Bearer test-key",
    }
    assert delivery.body == b'{"id": "a"}'


def test_config_reads_the_legacy_environment_variables() -> None:
    config = openmeter.Config.from_env({"OPENMETER_API_KEY": "k", "OPENMETER_API_ENDPOINT": "http://host/"})
    assert (config.api_key, config.url, config.event_type) == ("k", "http://host/api/v1/events", "litellm_tokens")
    with pytest.raises(ValueError, match="OPENMETER_API_KEY"):
        openmeter.Config.from_env({})
