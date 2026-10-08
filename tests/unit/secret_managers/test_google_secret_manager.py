import base64
import binascii
import json
from typing import Final

import google.auth
import httpx
import pytest
import respx
from google.auth.credentials import Credentials
from pydantic import ValidationError

import litellm
import litellm.proxy.proxy_server
from litellm.secret_managers.google_secret_manager import GoogleSecretManager
from litellm.secret_managers.main import get_secret
from litellm.types.secret_managers.main import KeyManagementSettings

PROJECT_ID: Final = "unit-test-project"
SECRET_NAME: Final = "UNIT_TEST_GOOGLE_SECRET"
ACCESS_URL: Final = (
    f"https://secretmanager.googleapis.com/v1/projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest:access"
)
SECRET_VALUE: Final = "s3cr3t-value"
ENCODED_SECRET: Final = base64.b64encode(SECRET_VALUE.encode()).decode("ascii")


class _StaticCredentials(Credentials):
    def refresh(self, request: object) -> None:
        self.token = "unit-test-token"


def _default_credentials(scopes: object = None) -> tuple[Credentials, str]:
    return _StaticCredentials(), PROJECT_ID


def _access_response(body: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(body).encode())


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> GoogleSecretManager:
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings())
    monkeypatch.setattr(litellm.vertex_chat_completion, "_credentials_project_mapping", {})
    monkeypatch.setattr(google.auth, "default", _default_credentials)
    monkeypatch.setenv("GOOGLE_SECRET_MANAGER_PROJECT_ID", PROJECT_ID)
    monkeypatch.delenv("GOOGLE_SECRET_MANAGER_ALWAYS_READ_SECRET_MANAGER", raising=False)
    monkeypatch.delenv("GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL", raising=False)
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_MOCK", raising=False)
    monkeypatch.delenv(SECRET_NAME, raising=False)
    return GoogleSecretManager()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"payload": {"data": ENCODED_SECRET}}, SECRET_VALUE),
        (
            {
                "name": f"projects/123456/secrets/{SECRET_NAME}/versions/7",
                "payload": {"data": ENCODED_SECRET, "dataCrc32c": "2932599711"},
            },
            SECRET_VALUE,
        ),
        ({"payload": {"data": ENCODED_SECRET, "labels": {"nested": [1, None]}}, "replication": [1, 2]}, SECRET_VALUE),
        ({"payload": {"data": "czNj\ncjN0 LXZhbHVl"}}, SECRET_VALUE),
        ({"payload": {"data": base64.b64encode("clé ☃".encode()).decode("ascii")}}, "clé ☃"),
        ({"payload": {"data": ""}}, ""),
    ],
)
@respx.mock
def test_secret_is_decoded_from_the_access_response(manager: GoogleSecretManager, body: object, expected: str) -> None:
    route: Final = respx.get(ACCESS_URL).mock(return_value=_access_response(body))

    secret: Final = manager.get_secret_from_google_secret_manager(SECRET_NAME)

    assert secret == expected
    assert type(secret) is str
    assert manager.cache.cache_dict[SECRET_NAME] is secret
    assert route.calls.last.request.headers["Authorization"] == "Bearer unit-test-token"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"name": f"projects/123456/secrets/{SECRET_NAME}/versions/7"},
        {"payload": {}},
        {"payload": {"dataCrc32c": "2932599711"}},
        {"payload": {"data": None}},
    ],
)
@respx.mock
def test_response_without_secret_data_reports_not_found_and_caches_the_miss(
    manager: GoogleSecretManager, body: object
) -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response(body))

    with pytest.raises(ValueError, match=rf"^secret {SECRET_NAME} not found in Google Secret Manager$") as raised:
        manager.get_secret_from_google_secret_manager(SECRET_NAME)

    assert type(raised.value) is ValueError
    assert manager.cache.cache_dict[SECRET_NAME] is None
    assert manager.get_secret_from_google_secret_manager(SECRET_NAME) is None


@pytest.mark.parametrize(
    "body",
    [
        [],
        [{"payload": {"data": ENCODED_SECRET}}],
        [["payload", {"data": ENCODED_SECRET}]],
        ENCODED_SECRET,
        7,
        None,
        True,
        {"payload": None},
        {"payload": []},
        {"payload": [["data", ENCODED_SECRET]]},
        {"payload": ENCODED_SECRET},
        {"payload": ""},
        {"payload": 0},
        {"payload": False},
    ],
)
@respx.mock
def test_response_that_is_not_a_json_object_is_rejected_without_caching(
    manager: GoogleSecretManager, body: object
) -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response(body))

    with pytest.raises(ValidationError) as raised:
        manager.get_secret_from_google_secret_manager(SECRET_NAME)

    assert [error["type"] for error in raised.value.errors()] == ["dict_type"]
    assert ENCODED_SECRET not in str(raised.value)
    assert SECRET_NAME not in manager.cache.cache_dict


@pytest.mark.parametrize("data", [5, 0, 1.5, True, False, [], [ENCODED_SECRET], {}, {"value": ENCODED_SECRET}])
@respx.mock
def test_secret_data_that_is_not_text_is_rejected_without_caching(manager: GoogleSecretManager, data: object) -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response({"payload": {"data": data}}))

    with pytest.raises(ValidationError) as raised:
        manager.get_secret_from_google_secret_manager(SECRET_NAME)

    assert [error["type"] for error in raised.value.errors()] == ["string_type"]
    assert ENCODED_SECRET not in str(raised.value)
    assert SECRET_NAME not in manager.cache.cache_dict


@pytest.mark.parametrize(
    "body",
    [
        [{"payload": {"data": ENCODED_SECRET}}] * 403 + [7],
        {"payload": [{"data": ENCODED_SECRET}] * 403 + [None]},
        {"payload": {"data": [ENCODED_SECRET] * 403 + [429]}},
        {"payload": {"data": {str(position): "Request timed out" for position in range(403, 430)}}},
    ],
)
@respx.mock
def test_rejection_of_a_long_response_names_no_position_or_content_from_it(
    manager: GoogleSecretManager, body: object
) -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response(body))

    with pytest.raises(ValidationError) as raised:
        manager.get_secret_from_google_secret_manager(SECRET_NAME)

    message: Final = str(raised.value)
    assert [error["loc"] for error in raised.value.errors()] == [()]
    assert [fragment for fragment in ("403", "429", "Request timed out", ENCODED_SECRET) if fragment in message] == []
    assert SECRET_NAME not in manager.cache.cache_dict


@pytest.mark.parametrize(
    ("data", "error_class"),
    [
        ("abc", binascii.Error),
        (base64.b64encode(b"\xff\xfe").decode("ascii"), UnicodeDecodeError),
        ("éééé", ValueError),
    ],
)
@respx.mock
def test_undecodable_secret_text_keeps_raising_the_decoder_error(
    manager: GoogleSecretManager, data: str, error_class: type[Exception]
) -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response({"payload": {"data": data}}))

    with pytest.raises(error_class) as raised:
        manager.get_secret_from_google_secret_manager(SECRET_NAME)

    assert type(raised.value) is error_class
    assert SECRET_NAME not in manager.cache.cache_dict


@pytest.mark.usefixtures("manager")
@respx.mock
def test_get_secret_reads_through_the_configured_manager() -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response({"payload": {"data": ENCODED_SECRET}}))

    assert get_secret(SECRET_NAME) == SECRET_VALUE


@pytest.mark.usefixtures("manager")
@pytest.mark.parametrize(
    "body",
    [
        [],
        {"payload": None},
        {"payload": ENCODED_SECRET},
        {"payload": {"data": 5}},
        {"payload": {"data": [ENCODED_SECRET]}},
        {"payload": {}},
        [{"payload": {"data": ENCODED_SECRET}}] * 403 + [7],
        {"payload": {"data": [ENCODED_SECRET] * 403 + [429]}},
        {"payload": "Request timed out"},
    ],
)
@respx.mock
def test_get_secret_falls_back_to_the_environment_when_the_response_is_unusable(
    monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    monkeypatch.setenv(SECRET_NAME, "from-environment")
    respx.get(ACCESS_URL).mock(return_value=_access_response(body))

    assert get_secret(SECRET_NAME) == "from-environment"


@pytest.mark.usefixtures("manager")
@pytest.mark.parametrize(
    "body",
    [
        [{"payload": {"data": ENCODED_SECRET}}] * 403 + [7],
        {"payload": None},
        {"payload": {"data": [ENCODED_SECRET] * 403 + [429]}},
        {"payload": {"data": 429}},
        {"payload": {}},
    ],
)
@respx.mock
def test_get_secret_returns_none_when_the_response_is_unusable_and_the_environment_has_no_value(body: object) -> None:
    respx.get(ACCESS_URL).mock(return_value=_access_response(body))

    assert get_secret(SECRET_NAME) is None
