from litellm.litellm_core_utils.internal_params import (
    LITELLM_INTERNAL_REQUEST_BODY_PARAMS,
    LiteLLMInternalParam,
    strip_internal_params_from_request_body,
)


def test_strips_every_registry_key():
    seeded = {param.value: "internal" for param in LiteLLMInternalParam}
    seeded.update({"temperature": 0.5, "max_tokens": 10})

    result = strip_internal_params_from_request_body(seeded)

    assert not (LITELLM_INTERNAL_REQUEST_BODY_PARAMS & result.keys())
    assert result == {"temperature": 0.5, "max_tokens": 10}



def test_keeps_unknown_provider_native_params():
    result = strip_internal_params_from_request_body({"anthropic_beta": "x", "top_k": 3})
    assert result == {"anthropic_beta": "x", "top_k": 3}

