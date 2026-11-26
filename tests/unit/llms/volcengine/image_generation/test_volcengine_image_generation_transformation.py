import json
from typing import Final

import httpx
import openai
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

API_BASE: Final = "https://ark.cn-beijing.volces.com"
IMAGES_URL: Final = f"{API_BASE}/api/v3/images/generations"
SEEDREAM_5_5: Final = "doubao-seedream-5-5-261028"

ARK_RESPONSE: Final[dict[str, object]] = {
    "created": 1784696685,
    "model": SEEDREAM_5_5,
    "data": [
        {
            "url": "https://ark.example/base.png",
            "size": "2048x2048",
            "output_format": "png",
            "background": "transparent",
            "z_index": 0,
        },
        {
            "url": "https://ark.example/parrot.png",
            "size": "2048x2048",
            "output_format": "png",
            "z_index": 1,
            "name": "parrot",
            "description": "a green parrot",
            "bounding_box": {"absolute": [347, 305, 642, 997], "normalized": [169, 305, 313, 997]},
        },
    ],
    "usage": {"generated_images": 2, "input_images": 1, "output_tokens": 32768, "total_tokens": 32768},
}


def _capturing_client(
    captured: dict[str, object],
    responses: list[httpx.Response] | None = None,
) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["x-custom"] = request.headers.get("x-custom")
        captured["body"] = json.loads(request.content)
        return responses[0] if responses else httpx.Response(200, json=ARK_RESPONSE)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_generation_sends_every_ark_param_and_maps_user_to_safety_identifier() -> None:
    captured: Final[dict[str, object]] = {}

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="split the parrot out of the base image",
        api_key="sk-ark",
        client=_capturing_client(captured),
        size="2K",
        response_format="url",
        image=["https://ark.example/input.png"],
        watermark=False,
        layer_decomposition=True,
        layer_decomposition_options={"max_images": 5},
        background="transparent",
        user="end-user-42",
        safety_identifier="explicit-id",
    )

    assert captured["url"] == IMAGES_URL
    assert captured["authorization"] == "Bearer sk-ark"
    assert captured["body"] == {
        "model": SEEDREAM_5_5,
        "prompt": "split the parrot out of the base image",
        "size": "2K",
        "response_format": "url",
        "image": ["https://ark.example/input.png"],
        "watermark": False,
        "layer_decomposition": True,
        "layer_decomposition_options": {"max_images": 5},
        "background": "transparent",
        "safety_identifier": "explicit-id",
    }
    assert response.data[0].url == "https://ark.example/base.png"
    assert response.data[1].provider_specific_fields == {
        "size": "2048x2048",
        "output_format": "png",
        "z_index": 1,
        "name": "parrot",
        "description": "a green parrot",
        "bounding_box": {"absolute": [347, 305, 642, 997], "normalized": [169, 305, 313, 997]},
    }
    assert (response.size, response.output_format, response.background) == ("2048x2048", "png", "transparent")
    assert response.usage is not None
    assert (response.usage.output_tokens, response.usage.total_tokens) == (32768, 32768)
    assert response.usage.model_dump()["input_images"] == 1
    assert response.usage.model_dump()["generated_images"] == 2


def test_generation_returns_base64_images() -> None:
    captured: Final[dict[str, object]] = {}
    body: Final[dict[str, object]] = {
        **ARK_RESPONSE,
        "data": [{"b64_json": "aW1n", "size": "1024x1024", "output_format": "jpeg"}],
    }

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=_capturing_client(captured, responses=[httpx.Response(200, json=body)]),
        response_format="b64_json",
    )

    assert captured["body"]["response_format"] == "b64_json"
    assert response.data[0].b64_json == "aW1n"
    assert response.data[0].url is None


def test_user_becomes_safety_identifier_when_no_explicit_id_is_given() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=_capturing_client(captured),
        user="end-user-42",
    )

    assert captured["body"]["safety_identifier"] == "end-user-42"
    assert "user" not in captured["body"]


def test_a_user_longer_than_ark_accepts_becomes_a_stable_fitting_safety_identifier() -> None:
    long_user: Final = "tenant-" + "x" * 120
    other_user: Final = "tenant-" + "y" * 120
    captured: Final[dict[str, object]] = {}

    def identifier_for(user: str) -> object:
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(captured),
            user=user,
        )
        return captured["body"]["safety_identifier"]

    first: Final = identifier_for(long_user)

    assert isinstance(first, str) and 0 < len(first) <= 64
    assert identifier_for(long_user) == first
    assert identifier_for(other_user) != first


def test_n_becomes_n_identical_ark_requests_and_n_is_never_sent_upstream() -> None:
    bodies: Final[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={**ARK_RESPONSE, "data": [{"url": "https://ark.example/one.png"}]})

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        n=3,
    )

    assert len(bodies) == 3
    assert all(body == {"model": SEEDREAM_5_5, "prompt": "a cat"} for body in bodies)
    assert len(response.data) == 3


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        (None, IMAGES_URL),
        ("https://gateway.example/ark", "https://gateway.example/ark/images/generations"),
        ("https://ark.example/api/v3", "https://ark.example/api/v3/images/generations"),
        ("https://ark.example/api/v3/", "https://ark.example/api/v3/images/generations"),
        ("https://ark.example/api/v3/images/generations", "https://ark.example/api/v3/images/generations"),
    ],
)
def test_generation_resolves_an_api_base_to_the_images_route(api_base: str | None, expected_url: str) -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        api_base=api_base,
        client=_capturing_client(captured),
    )

    assert captured["url"] == expected_url


@pytest.mark.parametrize(
    ("environment", "expected_url"),
    [
        ({"ARK_API_BASE": "https://ark-only.example/api/v3"}, "https://ark-only.example/api/v3/images/generations"),
        (
            {"VOLCENGINE_API_BASE": "https://volcengine.example/api/v3", "ARK_API_BASE": "https://ark.example/api/v3"},
            "https://volcengine.example/api/v3/images/generations",
        ),
    ],
)
def test_generation_and_edit_resolve_the_same_url_and_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected_url: str,
) -> None:
    monkeypatch.delenv("VOLCENGINE_API_BASE", raising=False)
    monkeypatch.delenv("ARK_API_BASE", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VOLCENGINE_API_KEY", "volcengine-key")
    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    generation: Final[dict[str, object]] = {}
    edit: Final[dict[str, object]] = {}

    litellm.image_generation(model=f"volcengine/{SEEDREAM_5_5}", prompt="a cat", client=_capturing_client(generation))
    litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        image="https://ark.example/input.png",
        client=_capturing_client(edit),
    )

    assert generation["url"] == expected_url
    assert edit["url"] == generation["url"]
    assert generation["authorization"] == "Bearer volcengine-key"
    assert edit["authorization"] == generation["authorization"]


def test_deployment_extra_headers_reach_ark_as_headers_and_never_the_body() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        extra_headers={"X-Custom": "kept"},
        client=_capturing_client(captured),
    )

    body: Final = captured["body"]
    assert isinstance(body, dict)
    assert captured["x-custom"] == "kept"
    assert "extra_headers" not in body
    assert captured["authorization"] == "Bearer sk-ark"


def test_a_caller_authorization_header_replaces_the_ark_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    captured: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        extra_headers={"authorization": "Bearer header-key"},
        client=_capturing_client(captured),
    )

    assert captured["authorization"] == "Bearer header-key"


def test_forwarded_client_headers_reach_ark_on_generation_and_edit() -> None:
    generation: Final[dict[str, object]] = {}
    edit: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        headers={"X-Custom": "forwarded"},
        client=_capturing_client(generation),
    )
    litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        image="https://ark.example/input.png",
        api_key="sk-ark",
        headers={"X-Custom": "forwarded"},
        client=_capturing_client(edit),
    )

    assert generation["x-custom"] == "forwarded"
    assert edit["x-custom"] == generation["x-custom"]
    assert generation["authorization"] == "Bearer sk-ark"


def test_the_global_litellm_api_key_authenticates_generation_and_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", "global-key")
    generation: Final[dict[str, object]] = {}
    edit: Final[dict[str, object]] = {}

    litellm.image_generation(model=f"volcengine/{SEEDREAM_5_5}", prompt="a cat", client=_capturing_client(generation))
    litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        image="https://ark.example/input.png",
        client=_capturing_client(edit),
    )

    assert generation["authorization"] == "Bearer global-key"
    assert edit["authorization"] == generation["authorization"]


@pytest.mark.parametrize("environment_key", ["VOLCENGINE_API_KEY", "ARK_API_KEY"])
def test_a_volcengine_environment_key_wins_over_the_global_litellm_api_key(
    monkeypatch: pytest.MonkeyPatch, environment_key: str
) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    monkeypatch.setenv(environment_key, "volcengine-key")
    monkeypatch.setattr(litellm, "api_key", "global-key-for-another-provider")
    generation: Final[dict[str, object]] = {}
    edit: Final[dict[str, object]] = {}

    litellm.image_generation(model=f"volcengine/{SEEDREAM_5_5}", prompt="a cat", client=_capturing_client(generation))
    litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        image="https://ark.example/input.png",
        client=_capturing_client(edit),
    )

    assert generation["authorization"] == "Bearer volcengine-key"
    assert edit["authorization"] == generation["authorization"]


def test_usage_reports_ark_output_tokens_as_completion_tokens() -> None:
    usage: Final = ARK_RESPONSE["usage"]
    assert isinstance(usage, dict)

    generation: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}", prompt="a cat", api_key="sk-ark", client=_capturing_client({})
    )
    edit: Final = litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        image="https://ark.example/input.png",
        api_key="sk-ark",
        client=_capturing_client({}),
    )

    assert generation.usage is not None
    assert generation.usage.completion_tokens == usage["output_tokens"]
    assert edit.usage is not None
    assert edit.usage.completion_tokens == generation.usage.completion_tokens


ONE_IMAGE: Final = {**ARK_RESPONSE, "data": [{"url": "https://ark.example/one.png"}]}
ARK_UNAVAILABLE: Final = {"error": {"code": "ServerOverloaded", "message": "try again"}}


def test_n_above_the_fan_out_cap_is_dropped_with_drop_params_on_generation_and_edit() -> None:
    bodies: Final[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ONE_IMAGE)

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    generation: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}", prompt="a cat", api_key="sk-ark", client=client, n=11, drop_params=True
    )
    edit: Final = litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        image="https://ark.example/input.png",
        api_key="sk-ark",
        client=client,
        n=11,
        drop_params=True,
    )

    assert len(bodies) == 2
    assert all("n" not in body for body in bodies)
    assert len(generation.data) == len(edit.data) == 1


@pytest.mark.parametrize(
    ("first_status", "retry_kwargs", "upstream_calls"),
    [
        (503, {}, 2),
        (503, {"max_retries": 0}, 1),
        (503, {"metadata": {"model_group": "seedream"}}, 1),
        (400, {}, 1),
    ],
    ids=["sdk-default-retries", "retries-off", "router-retries-instead", "client-error-not-retried"],
)
def test_a_transient_ark_failure_is_retried_unless_retries_are_off(
    first_status: int, retry_kwargs: dict[str, object], upstream_calls: int
) -> None:
    answers: Final = iter([httpx.Response(first_status, json=ARK_UNAVAILABLE), httpx.Response(200, json=ONE_IMAGE)])
    calls: Final[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return next(answers)

    def generate() -> litellm.ImageResponse:
        return litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
            **retry_kwargs,
        )

    if upstream_calls == 1:
        with pytest.raises((litellm.ServiceUnavailableError, litellm.BadRequestError)):
            generate()
    else:
        assert len(generate().data) == 1
    assert len(calls) == upstream_calls


@pytest.mark.asyncio
async def test_async_generation_fans_out_on_the_callers_async_client_and_retries_a_transient_failure() -> None:
    answers: Final = iter(
        [httpx.Response(503, json=ARK_UNAVAILABLE), httpx.Response(200, json=ONE_IMAGE), httpx.Response(200, json=ONE_IMAGE)]
    )
    bodies: Final[list[dict[str, object]]] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return next(answers)

    response: Final = await litellm.aimage_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(respond)),
        n=2,
    )

    assert len(bodies) == 3
    assert len(response.data) == 2


SUPPLIED_CLIENT_BASE: Final = "https://gateway.example/api/v3"


def _without_ark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ARK_API_KEY", "VOLCENGINE_API_KEY", "ARK_API_BASE", "VOLCENGINE_API_BASE"):
        monkeypatch.delenv(name, raising=False)


def _sent_with_the_supplied_clients_key_and_base(request: httpx.Request) -> bool:
    return (
        str(request.url) == f"{SUPPLIED_CLIENT_BASE}/images/generations"
        and request.headers.get("authorization") == "Bearer client-key"
    )


@pytest.mark.parametrize(
    ("client_max_retries", "call_max_retries", "unanswered"),
    [(0, None, 0), (1, 0, 1), (0, 1, 0)],
    ids=["litellm-default-retries-win", "call-retries-off-wins", "call-retries-once-wins"],
)
def test_a_supplied_openai_client_sends_generation_through_its_own_http_client_and_retries(
    monkeypatch: pytest.MonkeyPatch, client_max_retries: int, call_max_retries: int | None, unanswered: int
) -> None:
    _without_ark_env(monkeypatch)
    answers: Final = iter([httpx.Response(503, json=ARK_UNAVAILABLE), httpx.Response(200, json=ONE_IMAGE)])

    def respond(request: httpx.Request) -> httpx.Response:
        if not _sent_with_the_supplied_clients_key_and_base(request):
            return httpx.Response(401, json=ARK_UNAVAILABLE)
        return next(answers)

    client: Final = openai.OpenAI(
        api_key="client-key",
        base_url=SUPPLIED_CLIENT_BASE,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=client_max_retries,
    )

    def generate() -> litellm.ImageResponse:
        return litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}", prompt="a cat", client=client, max_retries=call_max_retries
        )

    if unanswered:
        with pytest.raises(litellm.ServiceUnavailableError):
            generate()
    else:
        assert len(generate().data) == 1
    assert sum(1 for _ in answers) == unanswered


@pytest.mark.asyncio
async def test_a_supplied_async_openai_client_sends_generation_through_its_own_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_ark_env(monkeypatch)

    async def respond(request: httpx.Request) -> httpx.Response:
        if not _sent_with_the_supplied_clients_key_and_base(request):
            return httpx.Response(401, json=ARK_UNAVAILABLE)
        return httpx.Response(200, json=ONE_IMAGE)

    response: Final = await litellm.aimage_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        client=openai.AsyncOpenAI(
            api_key="client-key",
            base_url=SUPPLIED_CLIENT_BASE,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )

    assert len(response.data) == 1


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("call_max_retries", [0, None], ids=["call-retries-off", "litellm-default-retries"])
async def test_a_connection_error_on_a_supplied_async_openai_client_is_never_retried_outside_it(
    monkeypatch: pytest.MonkeyPatch, call_max_retries: int | None
) -> None:
    _without_ark_env(monkeypatch)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    outside_the_client: Final = respx.post(f"{SUPPLIED_CLIENT_BASE}/images/generations").respond(200, json=ONE_IMAGE)
    answers: Final = iter(
        [httpx.ConnectError("the caller's proxy refused the connection"), httpx.Response(200, json=ONE_IMAGE)]
    )

    async def refuse_once(request: httpx.Request) -> httpx.Response:
        answer: Final = next(answers)
        if isinstance(answer, httpx.ConnectError):
            raise answer
        return answer

    async def generate() -> litellm.ImageResponse:
        return await litellm.aimage_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            client=openai.AsyncOpenAI(
                api_key="client-key",
                base_url=SUPPLIED_CLIENT_BASE,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse_once)),
                max_retries=0,
            ),
            max_retries=call_max_retries,
        )

    if call_max_retries is None:
        assert len((await generate()).data) == 1
    else:
        with pytest.raises(litellm.InternalServerError, match="the caller's proxy refused the connection"):
            await generate()
    assert not outside_the_client.called


def test_generation_surfaces_the_ark_error_message() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(
                captured,
                responses=[
                    httpx.Response(
                        400,
                        json={"error": {"code": "InvalidParameter", "message": "size 512x512 is below the minimum"}},
                    )
                ],
            ),
        )

    assert "size 512x512 is below the minimum" in str(exc_info.value)


def test_generation_without_an_ark_key_is_an_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            client=_capturing_client(captured),
        )

    assert "ARK_API_KEY" in str(exc_info.value)
    assert captured == {}


def test_generation_rejects_streaming_because_ark_streams_sse_outside_the_image_surface() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(captured),
            stream=True,
        )

    assert captured == {}


def test_generation_rejects_stream_sent_through_extra_body_so_no_image_goes_unbilled() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(captured),
            extra_body={"stream": True},
        )

    assert captured == {}


def test_generation_drops_stream_from_extra_body_when_drop_params_is_set() -> None:
    bodies: Final[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={**ARK_RESPONSE, "data": [{"url": "https://ark.example/one.png"}]})

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        n=2,
        extra_body={"stream": True, "watermark": False},
        drop_params=True,
    )

    assert bodies == [{"model": SEEDREAM_5_5, "prompt": "a cat", "watermark": False}] * 2
    assert len(response.data) == 2
