from litellm import Router


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "test",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ]
    )


def test_log_retry_keeps_history_request_local_and_omits_request_secrets():
    router = _router()
    first = router.log_retry(
        {
            "litellm_call_id": "call-1",
            "litellm_trace_id": "trace-1",
            "model": "test",
            "api_key": "FIRST_REQUEST_SECRET",
            "litellm_metadata": {"headers": {"authorization": "Bearer first-request-secret"}},
        },
        RuntimeError("first failure"),
    )
    second = router.log_retry(
        {
            "litellm_call_id": "call-2",
            "litellm_trace_id": "trace-2",
            "model": "test",
            "api_key": "SECOND_REQUEST_SECRET",
            "litellm_metadata": {"headers": {"authorization": "Bearer second-request-secret"}},
        },
        RuntimeError("second failure"),
    )

    first_history = first["litellm_metadata"]["previous_models"]
    second_history = second["litellm_metadata"]["previous_models"]

    assert [item["litellm_call_id"] for item in first_history] == ["call-1"]
    assert [item["litellm_call_id"] for item in second_history] == ["call-2"]
    assert first_history is not second_history
    serialized_history = repr(first_history + second_history)
    assert "FIRST_REQUEST_SECRET" not in serialized_history
    assert "SECOND_REQUEST_SECRET" not in serialized_history
    assert "first-request-secret" not in serialized_history
    assert "second-request-secret" not in serialized_history


def test_log_retry_caps_history_without_mutating_input_list():
    router = _router()
    original_history = [{"litellm_call_id": f"old-{index}"} for index in range(6)]
    kwargs = {
        "litellm_call_id": "current",
        "model": "test",
        "litellm_metadata": {"previous_models": original_history},
    }

    result = router.log_retry(kwargs, RuntimeError("current failure"))
    history = result["litellm_metadata"]["previous_models"]

    assert [item["litellm_call_id"] for item in history] == [
        "old-3",
        "old-4",
        "old-5",
        "current",
    ]
    assert len(original_history) == 6
    assert history is not original_history


def test_log_retry_redacts_and_bounds_exception_text():
    router = _router()
    secret = "sk-" + ("a" * 32)
    result = router.log_retry(
        {
            "litellm_call_id": "call-secret",
            "model": "test",
            "litellm_metadata": {},
        },
        RuntimeError(f"provider rejected api_key={secret} " + "x" * 300),
    )

    exception_string = result["litellm_metadata"]["previous_models"][0]["exception_string"]
    assert secret not in exception_string
    assert "REDACTED" in exception_string
    assert len(exception_string) <= 200


def test_log_retry_initializes_invalid_metadata_and_history_values():
    router = _router()
    result = router.log_retry(
        {
            "litellm_call_id": "call-invalid",
            "model": "test",
            "litellm_metadata": None,
        },
        RuntimeError("failure"),
    )

    assert [item["litellm_call_id"] for item in result["litellm_metadata"]["previous_models"]] == ["call-invalid"]
