"""Azure OpenAI image generation on the wire: URL, query, credential header and JSON body.

`/v1/images/generations` on an `azure/<deployment>` model goes through `litellm/images/main.py` (the
`custom_llm_provider == "azure"` branch) and `AzureChatCompletion.image_generation`
(`litellm/llms/azure/azure.py`), which posts with litellm's own httpx client, not the SDK:

- A dated `api_version` posts to `{api_base}/openai/deployments/{deployment}/images/generations?api-version=<v>`
  (`create_azure_base_url`), and the body carries no `model` field, because Azure routes by the URL and some
  image models refuse a `model` in the body (`azure_deployment_image_generation_json_body`, #26316).
- A `v1` / `latest` / `preview` version posts to `{resource}/openai/v1/images/generations?api-version=<v>`
  (`get_azure_v1_image_url`) and the body `model` is the deployment name, even when `base_model` is set.
- An `api_base` that already contains `/openai/deployments/<name>` is used as is.
- Credentials: `images/main.py` puts the key in an `api-key` header before the Entra check, so a key wins over
  an `azure_ad_token`; without a key the token is sent as `Authorization: Bearer`
  (`get_azure_request_auth_headers`). This is the opposite of chat, where the AzureOpenAI SDK prefers the token.

No spend is asserted here: image generation custom prices are currently lost (#52103).
"""

from __future__ import annotations

from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure3_support import PNG_B64, image_json
from integration.providers.dec1_azure_support import (
    API_KEY,
    DATED_API_VERSION,
    ENTRA_TOKEN,
    azure,
    azure_error,
    body_of,
    error_message,
    json_reply,
    marker,
    path_of,
    query_of,
)

IMAGE_DEPLOYMENT: Final = "dalle-dec1"
GPT_IMAGE_DEPLOYMENT: Final = "gpt-image-dec1"


def _ok(_: Request) -> Reply:
    return json_reply(image_json())


def test_a_dated_version_posts_to_the_deployment_images_route(gateway: Gateway) -> None:
    """An `azure/dalle-dec1` deployment with `api_version: 2024-10-21` generates one image.

    Azure must see exactly one `POST /openai/deployments/dalle-dec1/images/generations` whose query is exactly
    `{"api-version": "2024-10-21"}`, and the caller gets 200.
    """
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}") as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()}
        )
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (
            f"/openai/deployments/{IMAGE_DEPLOYMENT}/images/generations",
            {"api-version": DATED_API_VERSION},
        )


def test_the_deployment_route_body_carries_no_model_field(gateway: Gateway) -> None:
    """The caller sends `prompt`, `n: 1`, `size: 1024x1024` and `quality: hd` on the dated deployment route.

    The body Azure gets must be exactly `{"prompt", "n", "size", "quality"}` with the caller's values and no
    `model` key at all (`azure_deployment_image_generation_json_body` drops it on the deployment route).
    """
    prompt: Final = marker()
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}") as rig:
        response: Final = rig.gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": rig.model, "prompt": prompt, "n": 1, "size": "1024x1024", "quality": "hd"},
        )
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post()) == {"prompt": prompt, "n": 1, "size": "1024x1024", "quality": "hd"}


def test_the_deployment_key_is_sent_in_the_api_key_header(gateway: Gateway) -> None:
    """The deployment has a key and no Entra token.

    Azure must see `api-key: synthetic-azure-key` and no `Authorization` header, and the caller's virtual key
    never reaches Azure.
    """
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}") as rig:
        rig.gateway.request("POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()})
        request: Final = rig.one_post()
        assert request.headers.get("api-key") == API_KEY
        assert "authorization" not in request.headers


def test_an_entra_only_image_deployment_sends_a_bearer(gateway: Gateway) -> None:
    """The deployment has no key and `azure_ad_token: synthetic-entra-access-token`.

    `get_azure_request_auth_headers` resolves the token, so Azure must see
    `Authorization: Bearer synthetic-entra-access-token` and no `api-key` header. UNVERIFIED: assumes the proxy
    environment has no `AZURE_API_KEY` / `AZURE_OPENAI_API_KEY`, which `images/main.py` would read first.
    """
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}", api_key=None, azure_ad_token=ENTRA_TOKEN) as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()}
        )
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert request.headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "api-key" not in request.headers


def test_a_key_wins_over_an_entra_token_on_images(gateway: Gateway) -> None:
    """The deployment sets both a key and an `azure_ad_token`.

    `images/main.py` adds `api-key` to the headers first and `get_azure_request_auth_headers` keeps headers that
    already carry an Azure credential, so Azure must see only `api-key` and no `Authorization`. Chat does the
    opposite (round 1 `test_an_azure_ad_token_wins_over_a_key_when_both_are_configured`); a unified Rust
    credential resolver must decide this on purpose.
    """
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}", azure_ad_token=ENTRA_TOKEN) as rig:
        rig.gateway.request("POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()})
        request: Final = rig.one_post()
        assert (request.headers.get("api-key"), request.headers.get("authorization")) == (API_KEY, None)


def test_the_base64_image_and_revised_prompt_reach_the_caller(gateway: Gateway) -> None:
    """Azure answers `{"created": 1, "data": [{"b64_json": <png>, "revised_prompt": ...}]}`.

    The caller must get 200 with `data[0].b64_json` byte for byte and `data[0].revised_prompt` unchanged.
    """
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}") as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()}
        )
        assert response.status_code == 200, response.text
        image: Final = response.json()["data"][0]
        assert (image["b64_json"], image["revised_prompt"]) == (PNG_B64, "a synthetic lighthouse")


def test_the_v1_version_posts_to_the_openai_v1_images_route_with_the_deployment_as_model(gateway: Gateway) -> None:
    """An `azure/gpt-image-dec1` deployment with `api_version: v1`.

    `get_azure_v1_image_url` builds `/openai/v1/images/generations` and keeps the version in the query, so Azure
    must see that path with query exactly `{"api-version": "v1"}`, and the body `model` must be
    `gpt-image-dec1` because the v1 surface routes by body model. Note chat on `v1` sends no api-version at all.
    """
    with azure(gateway, _ok, model=f"azure/{GPT_IMAGE_DEPLOYMENT}", api_version="v1") as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()}
        )
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/images/generations", {"api-version": "v1"})
        assert body_of(request)["model"] == GPT_IMAGE_DEPLOYMENT


def test_the_v1_body_model_is_the_deployment_even_with_a_base_model(gateway: Gateway) -> None:
    """The v1 deployment also sets `base_model: azure/gpt-image-1`.

    The image data starts with `model: base_model`, but on the v1 route the deployment name replaces it, or
    Azure answers 404 DeploymentNotFound. The body `model` must be `gpt-image-dec1`, never `azure/gpt-image-1`.
    """
    with azure(
        gateway, _ok, model=f"azure/{GPT_IMAGE_DEPLOYMENT}", api_version="v1", base_model="azure/gpt-image-1"
    ) as rig:
        rig.gateway.request("POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()})
        assert body_of(rig.one_post())["model"] == GPT_IMAGE_DEPLOYMENT


def test_a_v1_api_base_with_a_deployment_path_is_cut_back_to_the_resource(gateway: Gateway) -> None:
    """`api_base` is `<peer>/openai/deployments/old-name?api-version=2024-02-01` and `api_version` is `v1`.

    `get_azure_v1_image_url` drops everything from `/openai` on and the stale api-version, so Azure must see
    `POST /openai/v1/images/generations` with query exactly `{"api-version": "v1"}`.
    """
    with azure(
        gateway,
        _ok,
        model=f"azure/{GPT_IMAGE_DEPLOYMENT}",
        api_version="v1",
        base_path="/openai/deployments/old-name?api-version=2024-02-01",
    ) as rig:
        rig.gateway.request("POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()})
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/images/generations", {"api-version": "v1"})


def test_an_api_base_already_naming_a_deployment_is_used_for_the_dated_route(gateway: Gateway) -> None:
    """`api_base` is `<peer>/openai/deployments/pinned-images` with a dated version.

    `create_azure_base_url` keeps an api_base that already contains `/openai/deployments/`, so Azure must see
    `POST /openai/deployments/pinned-images/images/generations?api-version=2024-10-21`, not the model name.
    """
    with azure(gateway, _ok, model=f"azure/{IMAGE_DEPLOYMENT}", base_path="/openai/deployments/pinned-images") as rig:
        rig.gateway.request("POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()})
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (
            "/openai/deployments/pinned-images/images/generations",
            {"api-version": DATED_API_VERSION},
        )


def test_an_azure_content_policy_refusal_on_images_is_a_400_with_azures_message(gateway: Gateway) -> None:
    """Azure answers 400 `{"error": {"code": "content_policy_violation", "message": "Your request was rejected..."}}`.

    The caller must get HTTP 400 and the error message must contain Azure's text. UNVERIFIED: the exact
    error `type` the image path maps to; only status and message are pinned.
    """
    refusal: Final = azure_error(400, "content_policy_violation", "Your request was rejected by the safety system")
    with azure(gateway, lambda _: refusal, model=f"azure/{IMAGE_DEPLOYMENT}", max_retries=0) as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/images/generations", {"model": rig.model, "prompt": marker()}
        )
        assert response.status_code == 400, response.text
        assert "Your request was rejected by the safety system" in error_message(response)
