from queue import SimpleQueue
from typing import Final

import pytest
from fastapi import HTTPException

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.route_enable import RouteFeature, RouteSetting
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.route_enable_adapter import RouteEnableReaders, check_route_enabled


@pytest.mark.parametrize("route", ("/models", "/chat/completions"))
def test_enterprise_class_exemptions_preserve_enabled_route_policy(route: str) -> None:
    class EnterpriseRoutes:
        LLM_API_EXEMPT_ROUTES: Final = ("/models",)

    readers: Final = RouteEnableReaders(
        enterprise=lambda: EnterpriseRoutes,
        setting=lambda _: RouteSetting(True, True),
        secret=lambda _: True,
    )
    if route == "/models":
        assert check_route_enabled(route, readers=readers) is None
    else:
        with pytest.raises(HTTPException) as error:
            check_route_enabled(route, readers=readers)
        assert (error.value.status_code, error.value.detail) == (
            403,
            "LLM API routes are disabled for this instance.",
        )


def test_missing_optional_class_exemptions_preserve_ignored_read_failure() -> None:
    readers: Final = RouteEnableReaders(enterprise=lambda: object())
    assert check_route_enabled("/chat/completions", readers=readers) is None


@pytest.mark.parametrize(
    ("route", "expected"),
    (
        ("/unknown", ("import",)),
        ("/models", ("import", "exemptions")),
        ("/chat/completions", ("import", "exemptions", "Llm", "secret-Llm")),
        ("/key/generate", ("import", "Admin", "secret-Admin")),
    ),
)
def test_optional_route_reads_are_lazy(route: str, expected: tuple[str, ...]) -> None:
    events: Final[SimpleQueue[str]] = SimpleQueue()

    def enterprise() -> object:
        events.put("import")
        return object()

    def setting(feature: RouteFeature) -> RouteSetting:
        events.put(feature)
        return RouteSetting(False, False)

    def secret(feature: RouteFeature) -> bool:
        events.put(f"secret-{feature}")
        return False

    def exemptions(_enterprise: object) -> tuple[str, ...]:
        events.put("exemptions")
        return ("/models",)

    check_route_enabled(route, readers=RouteEnableReaders(enterprise, setting, secret, exemptions))
    assert tuple(events.get() for _ in range(events.qsize())) == expected


@pytest.mark.parametrize(("route", "label"), (("/key/generate", "ADMIN"), ("/chat/completions", "LLM API")))
@pytest.mark.parametrize("premium", (False, True))
def test_optional_route_denial_envelopes(route: str, label: str, premium: bool) -> None:
    readers: Final = RouteEnableReaders(
        enterprise=lambda: object(),
        setting=lambda _: RouteSetting(True, premium),
        secret=lambda _: True,
        exemptions=lambda _: (),
    )
    with pytest.raises(HTTPException) as error:
        check_route_enabled(route, readers=readers)
    if premium:
        disabled: Final = "Management" if label == "ADMIN" else label
        assert (error.value.status_code, error.value.detail) == (
            403,
            f"{disabled} routes are disabled for this instance.",
        )
    else:
        assert (error.value.status_code, error.value.detail) == (
            500,
            f"\U0001f6a8\U0001f6a8\U0001f6a8 DISABLING {label} ENDPOINTS is an Enterprise feature\n\U0001f6a8 {CommonProxyErrors.not_premium_user.value}",
        )


@pytest.mark.parametrize(
    "error",
    (
        ImportError("optional package"),
        RuntimeError("settings unavailable"),
        HTTPException(418, "read failure"),
        NativeAuthRuntimeError("terminal native failure"),
    ),
)
def test_optional_import_failure_policy(error: Exception) -> None:
    def enterprise() -> object:
        raise error

    readers: Final = RouteEnableReaders(enterprise=enterprise)
    if isinstance(error, (HTTPException, NativeAuthRuntimeError)):
        with pytest.raises(type(error)) as caught:
            check_route_enabled("/key/generate", readers=readers)
        assert caught.value is error
    else:
        assert check_route_enabled("/key/generate", readers=readers) is None
