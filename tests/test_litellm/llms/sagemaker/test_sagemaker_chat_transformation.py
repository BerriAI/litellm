"""Tests for ``SagemakerChatConfig`` Inference Component header injection
(Linear LIT-2853).

Bug: ``sagemaker_chat`` did not honor ``model_id`` the way the legacy
``sagemaker`` (completion) handler does, so callers targeting a SageMaker
endpoint that hosts multiple Inference Components had to pass
``extra_headers={"X-Amzn-SageMaker-Inference-Component": ...}`` by hand. The
fix moves ``model_id`` from either ``optional_params`` (in
``validate_environment``) or the merged ``request_data`` (in
``sign_request``) into the ``X-Amzn-SageMaker-Inference-Component`` header so
the behavior matches the completion path.

We exercise both hooks so a regression in either call shape (``model_id``
arriving via ``optional_params`` vs. via ``extra_body``) is caught.
"""
from unittest.mock import patch

from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig


# ----------------------------------------------------------------------
# validate_environment path (model_id arrives via optional_params)
# ----------------------------------------------------------------------


def test_validate_env_model_id_added_as_inference_component_header():
    """When ``model_id`` is in ``optional_params``, ``validate_environment``
    moves it into the SageMaker Inference Component header and removes it
    from ``optional_params`` so it does not also leak into the request body
    via ``transform_request`` (which spreads ``**optional_params``)."""
    config = SagemakerChatConfig()
    optional_params = {"model_id": "claude-multi-tenant-1", "temperature": 0.5}
    headers = {"Content-Type": "application/json"}
    out = config.validate_environment(
        headers=headers,
        model="my-endpoint",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
    )
    assert out["X-Amzn-SageMaker-Inference-Component"] == "claude-multi-tenant-1"
    assert out["Content-Type"] == "application/json"
    assert "model_id" not in optional_params
    assert optional_params == {"temperature": 0.5}


def test_validate_env_no_model_id_no_header_added():
    config = SagemakerChatConfig()
    optional_params = {"temperature": 0.5}
    headers = {"Content-Type": "application/json"}
    out = config.validate_environment(
        headers=headers,
        model="my-endpoint",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
    )
    assert "X-Amzn-SageMaker-Inference-Component" not in out
    assert out == headers
    assert optional_params == {"temperature": 0.5}


def test_validate_env_model_id_none_no_header_added():
    """An explicit ``model_id=None`` is treated the same as absent (no
    header is injected) but is still popped so the key cannot leak."""
    config = SagemakerChatConfig()
    optional_params = {"model_id": None, "temperature": 0.5}
    out = config.validate_environment(
        headers={},
        model="my-endpoint",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
    )
    assert "X-Amzn-SageMaker-Inference-Component" not in out
    assert "model_id" not in optional_params


# ----------------------------------------------------------------------
# sign_request path (model_id arrives via extra_body -> request_data)
# ----------------------------------------------------------------------


def _stub_sign_passthrough(self, **kwargs):
    """Replace `_sign_request` with a no-op SigV4 stub so the
    `sign_request` tests don't need real AWS credentials."""
    return kwargs["headers"], None


def test_sign_request_model_id_in_request_data_moves_to_header():
    """When ``model_id`` lives in ``request_data`` (which happens when the
    caller passes ``extra_body={"model_id": ...}`` -- ``BaseLLMHTTPHandler``
    merges ``extra_body`` into the request body *after* ``transform_request``
    runs), ``sign_request`` moves it into the header and pops it from the
    body so SigV4 signs the right payload."""
    config = SagemakerChatConfig()
    headers = {"Content-Type": "application/json"}
    request_data = {
        "model": "my-endpoint",
        "messages": [{"role": "user", "content": "hi"}],
        "model_id": "component-from-extra-body",
    }
    optional_params = {}

    with patch.object(
        SagemakerChatConfig, "_sign_request", _stub_sign_passthrough
    ):
        out_headers, signed = config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base="https://example/endpoints/my-endpoint/invocations",
        )

    assert out_headers["X-Amzn-SageMaker-Inference-Component"] == "component-from-extra-body"
    assert "model_id" not in request_data
    assert request_data["model"] == "my-endpoint"
    assert request_data["messages"] == [{"role": "user", "content": "hi"}]


def test_sign_request_model_id_in_optional_params_moves_to_header():
    """``sign_request`` also handles the ``optional_params`` shape (defense
    in depth)."""
    config = SagemakerChatConfig()
    headers = {}
    request_data = {"model": "my-endpoint", "messages": []}
    optional_params = {"model_id": "component-from-optional-params"}

    with patch.object(
        SagemakerChatConfig, "_sign_request", _stub_sign_passthrough
    ):
        out_headers, _ = config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base="https://example/endpoints/my-endpoint/invocations",
        )

    assert out_headers["X-Amzn-SageMaker-Inference-Component"] == "component-from-optional-params"
    assert "model_id" not in optional_params


def test_sign_request_no_model_id_no_header():
    """When no ``model_id`` is present in either shape, no header is added
    and the body is forwarded unchanged."""
    config = SagemakerChatConfig()
    headers = {}
    request_data = {"model": "my-endpoint", "messages": []}
    optional_params = {}

    with patch.object(
        SagemakerChatConfig, "_sign_request", _stub_sign_passthrough
    ):
        out_headers, _ = config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base="https://example/endpoints/my-endpoint/invocations",
        )

    assert "X-Amzn-SageMaker-Inference-Component" not in out_headers
    assert request_data == {"model": "my-endpoint", "messages": []}


def test_sign_request_request_data_wins_over_optional_params():
    """If ``model_id`` is present in BOTH ``request_data`` (from
    ``extra_body``) AND ``optional_params``, ``request_data`` wins -- it is
    the more specific, last-set source ahead of signing -- and the key is
    cleaned out of both."""
    config = SagemakerChatConfig()
    headers = {}
    request_data = {
        "model": "my-endpoint",
        "messages": [],
        "model_id": "component-from-extra-body",
    }
    optional_params = {"model_id": "component-from-optional-params"}

    with patch.object(
        SagemakerChatConfig, "_sign_request", _stub_sign_passthrough
    ):
        out_headers, _ = config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base="https://example/endpoints/my-endpoint/invocations",
        )

    assert out_headers["X-Amzn-SageMaker-Inference-Component"] == "component-from-extra-body"
    assert "model_id" not in request_data
    assert "model_id" not in optional_params
