"""Azure OpenAI `api_version` handling and its defaults on `/v1/chat/completions`.

Which `api_version` a deployment carries decides both the URL and which parameters are allowed:

A dated or dated-preview version (`2024-10-21`, `2025-04-01-preview`) uses the classic
`/openai/deployments/{deployment}/chat/completions?api-version=<v>` route through `AzureOpenAI`.
`v1`, `latest` and `preview` select Azure's v1 surface (`BaseAzureLLM.is_azure_v1_api_version`,
`litellm/llms/azure/common_utils.py:1160`): litellm then builds a plain `OpenAI` client whose `base_url` is
`{api_base}/openai/v1/` (`common_utils.py:804`), so the call is `POST /openai/v1/chat/completions` with the
deployment name in the body, no `api-version` query and the key as `Authorization: Bearer`. The opt-out value
`none` (any case) maps to `v1` for the chat client (`common_utils.py:1174`). With no version at all the call
falls back to `AZURE_API_VERSION` and then `litellm.AZURE_DEFAULT_API_VERSION = "2025-02-01-preview"`
(`litellm/main.py:1525`).

Older dated versions also gate parameters before any request leaves the proxy
(`gpt_transformation.py:238`): `tool_choice` needs `2023-12-01` or later, `tool_choice: "required"` is refused
on 2024 versions up to May, and `json_schema` response formats on versions before 2024-08 become a forced
`json_tool_call` tool. Those refusals are `UnsupportedParamsError` with status 400 and Azure is never called.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    API_KEY,
    DEFAULT_AZURE_API_VERSION,
    DEPLOYMENT,
    PREVIEW_API_VERSION,
    WEATHER_TOOL,
    azure,
    body_of,
    chat_reply,
    error_message,
    path_of,
    query_of,
)


def _ok(_: Request) -> Reply:
    return chat_reply("pong")


def test_a_preview_dated_version_is_sent_unchanged_on_the_deployment_route(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2025-04-01-preview`.

    Azure must see `POST /openai/deployments/gpt-4o-dec1/chat/completions` with query exactly
    `{"api-version": "2025-04-01-preview"}`.
    """
    with azure(gateway, _ok, api_version=PREVIEW_API_VERSION) as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert path_of(request) == f"/openai/deployments/{DEPLOYMENT}/chat/completions"
        assert query_of(request) == {"api-version": PREVIEW_API_VERSION}


def test_no_configured_version_falls_back_to_the_litellm_default(gateway: Gateway) -> None:
    """The deployment has no `api_version` and the shared proxy has no `AZURE_API_VERSION` in its environment.

    `_complete_azure` resolves the version as call value, `litellm.api_version`, `AZURE_API_VERSION`, then
    `litellm.AZURE_DEFAULT_API_VERSION` (`litellm/main.py:1525`), so Azure must see `api-version=2025-02-01-preview`.
    UNVERIFIED: assumes the CI environment does not export `AZURE_API_VERSION`.
    """
    with azure(gateway, _ok, api_version=None) as rig:
        assert rig.chat().status_code == 200
        assert query_of(rig.one_post()) == {"api-version": DEFAULT_AZURE_API_VERSION}


@pytest.mark.parametrize("version", ["v1", "latest", "preview"])
def test_a_v1_family_version_posts_to_the_openai_v1_route(gateway: Gateway, version: str) -> None:
    """The deployment sets `api_version` to `v1`, `latest` or `preview`.

    litellm builds a plain OpenAI client on `{api_base}/openai/v1/` (`common_utils.py:804`), so Azure must see
    `POST /openai/v1/chat/completions`, never a `/openai/deployments/...` path, and the body `model` must be
    the deployment name `gpt-4o-dec1`.
    """
    with azure(gateway, _ok, api_version=version) as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert path_of(request) == "/openai/v1/chat/completions"
        assert body_of(request)["model"] == DEPLOYMENT


def test_the_v1_route_sends_no_api_version_query(gateway: Gateway) -> None:
    """The deployment sets `api_version: v1`.

    The v1 client's `default_query` is the api_base query minus `api-version` (`common_utils.py:827`), so the
    request Azure receives must have no query string at all.
    """
    with azure(gateway, _ok, api_version="v1") as rig:
        assert rig.chat().status_code == 200
        assert query_of(rig.one_post()) == {}


def test_the_v1_route_sends_the_key_as_a_bearer_token(gateway: Gateway) -> None:
    """The deployment sets `api_version: v1` and a key.

    The plain OpenAI client sends its `api_key` as `Authorization: Bearer synthetic-azure-key`, which Azure's
    v1 surface accepts. A port that sends `api-key` here instead still works on Azure, but must not send the
    key in both headers. UNVERIFIED: that the OpenAI client adds no `api-key` header.
    """
    with azure(gateway, _ok, api_version="v1") as rig:
        assert rig.chat().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("authorization") == f"Bearer {API_KEY}"
        assert "api-key" not in headers, headers


@pytest.mark.parametrize("version", ["none", "None"])
def test_the_api_version_opt_out_uses_the_v1_route(gateway: Gateway, version: str) -> None:
    """The deployment sets `api_version` to the opt-out value `none` in any casing.

    `chat_client_api_version` maps it to `v1` (`common_utils.py:1174`), so Azure must see
    `POST /openai/v1/chat/completions` with no `api-version` query, never `api-version=none`.
    """
    with azure(gateway, _ok, api_version=version) as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert path_of(request) == "/openai/v1/chat/completions"
        assert "api-version" not in query_of(request)


def test_an_api_base_ending_in_openai_v1_is_not_doubled(gateway: Gateway) -> None:
    """`api_base` is `<peer>/openai/v1` and `api_version` is `v1`.

    The v1 root keeps a base that already ends in `/openai/v1` (`common_utils.py:830`), so Azure must see
    exactly `POST /openai/v1/chat/completions`, never `/openai/v1/openai/v1/...`.
    """
    with azure(gateway, _ok, api_version="v1", base_path="/openai/v1") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.one_post()) == "/openai/v1/chat/completions"


def test_an_api_base_ending_in_openai_gets_v1_appended_once(gateway: Gateway) -> None:
    """`api_base` is `<peer>/openai` and `api_version` is `v1`.

    The v1 root strips the trailing `/openai` before adding `/openai/v1` (`common_utils.py:831`), so Azure must
    see `POST /openai/v1/chat/completions`, never `/openai/openai/v1/...`.
    """
    with azure(gateway, _ok, api_version="v1", base_path="/openai") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.one_post()) == "/openai/v1/chat/completions"


def test_tool_choice_on_an_api_version_before_2023_12_is_refused_before_azure(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2023-07-01-preview`; the caller sends tools and `tool_choice: "auto"`.

    Without drop_params `map_openai_params` raises `UnsupportedParamsError` (`gpt_transformation.py:250`):
    the caller must get HTTP 400 whose message contains `Azure does not support 'tool_choice', for
    api_version=2023-07-01-preview`, and Azure must receive no request.
    """
    with azure(gateway, _ok, api_version="2023-07-01-preview") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="auto")
        assert response.status_code == 400, response.text
        assert "Azure does not support 'tool_choice', for api_version=2023-07-01-preview" in error_message(response)
        assert rig.posts() == ()


def test_tool_choice_required_on_a_may_2024_api_version_is_refused(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2024-05-01-preview`; the caller sends tools and `tool_choice: "required"`.

    The caller must get HTTP 400 whose message contains `Azure does not support 'required' as a tool_choice
    param, for api_version=2024-05-01-preview`, and Azure must receive no request.
    """
    with azure(gateway, _ok, api_version="2024-05-01-preview") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="required")
        assert response.status_code == 400, response.text
        assert "Azure does not support 'required' as a tool_choice param, for api_version=2024-05-01-preview" in (
            error_message(response)
        )
        assert rig.posts() == ()


def test_tool_choice_required_on_a_june_2024_api_version_is_forwarded(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2024-06-01`; the caller sends tools and `tool_choice: "required"`.

    June 2024 is past the cutoff, so Azure must receive `tool_choice: "required"` and the caller gets HTTP 200.
    """
    with azure(gateway, _ok, api_version="2024-06-01") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="required")
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["tool_choice"] == "required"


def test_tool_choice_required_on_an_old_version_is_dropped_when_the_request_drops_params(gateway: Gateway) -> None:
    """Same `2024-05-01-preview` deployment, but the request sets `drop_params: true`.

    The refusal is skipped (`gpt_transformation.py:257`): Azure must receive the tools but no `tool_choice`, and
    the caller gets HTTP 200.
    """
    with azure(gateway, _ok, api_version="2024-05-01-preview") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="required", drop_params=True)
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
        assert "tool_choice" not in body, body
        assert body["tools"] == [WEATHER_TOOL]


def test_json_schema_on_a_pre_august_2024_version_becomes_a_forced_tool(gateway: Gateway) -> None:
    """A gpt-4o deployment with `api_version: 2024-06-01` and a `json_schema` response format.

    Response formats need 2024-08 or later (`litellm/types/llms/azure.py:3`), so Azure must receive no
    `response_format` and instead a single `json_tool_call` tool with the schema as its parameters.
    """
    schema: Final = {"type": "object", "properties": {"answer": {"type": "string"}}}
    with azure(gateway, _ok, api_version="2024-06-01") as rig:
        response: Final = rig.chat(
            response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}}
        )
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
        assert "response_format" not in body, body
        assert [tool["function"]["name"] for tool in body["tools"]] == ["json_tool_call"]  # pyright: ignore[reportIndexIssue, reportOptionalIterable, reportGeneralTypeIssues]


def test_a_json_object_response_format_on_a_pre_august_2024_version_is_dropped(gateway: Gateway) -> None:
    """`response_format: {"type": "json_object"}` on a `2024-06-01` deployment.

    The tool conversion only applies when a schema is present (`base_llm/chat/transformation.py:272`); with no
    schema and an unsupported version nothing is added and the format is dropped, so Azure must receive neither
    `response_format` nor a `json_tool_call` tool. UNVERIFIED: the drop is silent rather than a 400.
    """
    with azure(gateway, _ok, api_version="2024-06-01") as rig:
        response: Final = rig.chat(response_format={"type": "json_object"})
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
        assert "response_format" not in body, body
        assert "tools" not in body, body
