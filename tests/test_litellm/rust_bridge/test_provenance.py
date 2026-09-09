from typing import Final

from litellm.rust_bridge.provenance import (
    RUST_RESPONSE_HEADER,
    has_rust_response_marker,
    mark_rust_response,
    rust_response_hidden_params,
    rust_stream_hidden_params,
)


class HiddenParamsCarrier:
    def __init__(self, hidden_params: dict[str, object]) -> None:
        self._hidden_params: dict[str, object] = hidden_params

    def hidden_params(self) -> dict[str, object]:
        return self._hidden_params


def test_marker_merges_dict_response_metadata_without_mutating_the_template() -> None:
    template: Final = {
        "response_cost": 1.25,
        "additional_headers": {"llm-provider-request-id": "req-1"},
    }
    response: Final = {"_hidden_params": template}

    assert mark_rust_response(response) is response
    assert mark_rust_response(response) is response
    assert response["_hidden_params"] == {
        "response_cost": 1.25,
        "additional_headers": {
            "llm-provider-request-id": "req-1",
            RUST_RESPONSE_HEADER: "true",
        },
    }
    assert template == {
        "response_cost": 1.25,
        "additional_headers": {"llm-provider-request-id": "req-1"},
    }


def test_marker_merges_object_response_metadata() -> None:
    response: Final = HiddenParamsCarrier(
        {"additional_headers": {"llm-provider-request-id": "req-2"}}
    )

    assert mark_rust_response(response) is response
    assert has_rust_response_marker(response)
    assert response.hidden_params()["additional_headers"] == {
        "llm-provider-request-id": "req-2",
        RUST_RESPONSE_HEADER: "true",
    }


def test_hidden_param_factories_return_independent_values() -> None:
    first: Final = rust_response_hidden_params()
    second: Final = rust_response_hidden_params()
    stream: Final = rust_stream_hidden_params({"llm-provider-request-id": "req-3"})

    assert first is not second
    assert first["additional_headers"] is not second["additional_headers"]
    assert stream == {
        "additional_headers": {
            "llm-provider-request-id": "req-3",
            RUST_RESPONSE_HEADER: "true",
        }
    }
