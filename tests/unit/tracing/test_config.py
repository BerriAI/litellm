import pytest


@pytest.mark.parametrize(
    "settings,environ,enabled",
    (
        (None, {}, False),
        ({"store": {"type": "clickhouse"}}, {}, False),
        ({"store": {"type": "lens"}}, {}, True),
        ({"store": "lens"}, {}, False),
        (None, {"LITELLM_LENS_URL": "http://lens"}, True),
    ),
)
def test_lens_enablement_requires_its_service_or_an_explicit_lens_store(
    settings: object, environ: dict[str, str], enabled: bool
) -> None:
    from litellm.tracing.config import is_lens_tracing_enabled

    assert is_lens_tracing_enabled(settings, environ) is enabled
