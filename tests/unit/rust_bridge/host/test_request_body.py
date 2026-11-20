import pytest

import litellm
from litellm.router import Router
from litellm.rust_bridge.host import request_body


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ('{"langfuse_host": "https://attacker.example"}', {"langfuse_host": "https://attacker.example"}),
        ('["langfuse_host"]', None),
        ('"langfuse_host"', None),
        ("not json", None),
    ),
)
def test_only_a_json_object_decodes_to_a_mapping(text: str, expected: dict[str, object] | None) -> None:
    assert request_body.decode_mapping(text) == expected


def test_a_mapping_the_proxy_would_decode_is_decoded_for_the_check_too() -> None:
    decoded = request_body.decode_mapping('{"sampling_rate": NaN, "langfuse_host": "https://attacker.example"}')

    assert decoded is not None
    assert decoded["langfuse_host"] == "https://attacker.example"


def test_form_metadata_builds_the_mapping_its_bracket_keys_spell() -> None:
    body = {"model": "whisper-1", "metadata[langfuse_host]": "https://attacker.example", "metadata[tags][team]": "a"}

    assert request_body.form_metadata(body, "metadata[") == {
        "langfuse_host": "https://attacker.example",
        "tags": {"team": "a"},
    }


@pytest.mark.parametrize(
    ("value", "allowed_hosts", "expected"),
    (
        ("huggingface/https://trusted.example/v1", ["trusted.example"], True),
        ("huggingface/https://attacker.example/v1", ["trusted.example"], False),
        ("huggingface/http://attacker.example/v1", [], False),
        ("huggingface/HTTPS://attacker.example/v1", [], False),
        ("https://trusted.example/v1,https://attacker.example/v1", ["trusted.example"], False),
        ("gpt-4o", [], True),
    ),
)
def test_a_url_destination_is_allowed_only_when_every_url_it_names_is_on_a_listed_host(
    monkeypatch: pytest.MonkeyPatch, value: str, allowed_hosts: list[str], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", allowed_hosts)

    assert request_body.url_destination_allowed(value) is expected


@pytest.fixture
def router() -> Router:
    def deployment(name: str, model: str, opt_in: list[object] | None) -> dict[str, object]:
        opted = {} if opt_in is None else {"configurable_clientside_auth_params": opt_in}
        return {"model_name": name, "litellm_params": {"model": model, "api_key": "k", **opted}}

    return Router(
        model_list=[
            deployment("primary", "azure/gpt-4o", ["client_id", {"api_base": "^https://allowed\\..*$"}]),
            deployment("backup", "openai/gpt-4o", None),
            deployment("last", "openai/gpt-4o-mini", []),
        ],
        fallbacks=[{"primary": ["backup"]}, {"backup": ["last"]}],
        model_group_alias={"nickname": "primary"},
    )


def test_a_group_reports_each_deployments_provider_and_opt_in(router: Router) -> None:
    assert request_body.group_deployments(router, "primary", None) == (
        ("primary", "azure", (("client_id",), ("^https://allowed\\..*$",))),
    )
    assert request_body.group_deployments(router, "backup", None) == (("backup", "openai", None),)
    assert request_body.group_deployments(router, "last", None) == (("last", "openai", ((), ())),)
    assert request_body.group_deployments(router, "unknown", None) == ()


def test_fallbacks_follow_the_routers_chain(router: Router) -> None:
    assert request_body.fallbacks(router, "primary") == ("backup", "last")
    assert request_body.fallbacks(router, "last") == ()


def test_an_alias_names_its_group_and_a_group_names_none(router: Router) -> None:
    assert request_body.alias(router, "nickname") == "primary"
    assert request_body.alias(router, "primary") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    (("https://allowed.example/v1", True), ("https://attacker.example/v1", False), ("^https://allowed\\..*$", True)),
)
def test_an_api_base_matches_its_pattern_from_the_start_or_by_equality(value: str, expected: bool) -> None:
    assert request_body.api_base_matches("^https://allowed\\..*$", value) is expected
