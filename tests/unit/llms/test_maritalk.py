import httpx

from litellm.llms.maritalk import MaritalkConfig, MaritalkError


def test_map_openai_params_keeps_maritalk_params_and_drops_other_openai_params():
    mapped = MaritalkConfig().map_openai_params(
        non_default_params={"temperature": 0.3, "top_k": 4, "logit_bias": {"1": 2}},
        optional_params={},
        model="sabia-3",
        drop_params=False,
    )

    assert mapped == {"temperature": 0.3, "top_k": 4}


def test_get_error_class_returns_a_maritalk_error_with_status_message_and_headers():
    headers = httpx.Headers({"retry-after": "3"})

    error = MaritalkConfig().get_error_class(error_message="rate limited", status_code=429, headers=headers)

    assert type(error) is MaritalkError
    assert (error.status_code, error.message, error.headers) == (429, "rate limited", headers)


def test_constructor_arguments_become_config_of_the_constructed_class_only():
    class ScopedMaritalkConfig(MaritalkConfig):
        pass

    ScopedMaritalkConfig(temperature=0.0, max_tokens=128, stop=["\n"])

    assert ScopedMaritalkConfig.get_config() == {"temperature": 0.0, "max_tokens": 128, "stop": ["\n"]}
    assert MaritalkConfig.get_config() == {}
