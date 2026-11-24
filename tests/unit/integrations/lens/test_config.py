import pytest


@pytest.mark.parametrize(
    "settings,environ,enabled",
    (
        (None, {}, False),
        ({"store": {"type": "clickhouse"}}, {}, False),
        ({"store": {"type": "lens"}}, {}, True),
        ({"store": "lens"}, {}, False),
        (None, {"LITELLM_LENS_URL": "http://lens"}, True),
        (None, {"LITELLM_LENS_PUBLIC_URL": "http://lens"}, False),
        (None, {"LITELLM_LENS_SERVICE_TOKEN": "configured-service-token"}, False),
        (None, {"LENS_GATEWAY_SECRET": "configured-signing-secret"}, False),
    ),
)
def test_lens_enablement_requires_its_service_or_an_explicit_lens_store(
    settings: object, environ: dict[str, str], enabled: bool
) -> None:
    from litellm.integrations.lens.config import LensController, is_lens_tracing_enabled

    assert is_lens_tracing_enabled(settings, environ) is enabled
    assert LensController.is_lens_setup(settings, environ) is enabled
