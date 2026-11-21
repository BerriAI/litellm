"""How `model_list` entries become deployments: declared ids, repeated ids, organization lists, and how
wildcard names are matched.

Each `model_list` entry is built into one deployment by the Rust router core
(`litellm/rust/crates/router-core/src/build.rs`) through the Python host
(`litellm/router_rust/surface/registry.py`). An entry keeps a declared `model_info.id`; two entries may
declare the same id (only PTU cost attribution refuses that), both are built, and the id index points at
the later one (`registry.rs` `push_built`). An entry whose `organization` is a list becomes one deployment
per organization, all sharing the entry's id. A wildcard `model_name` is matched anchored at the start
only, and when several patterns match, the more specific (longer) pattern wins (`pattern_specificity`),
while an exact name always beats any pattern.

These are config-loading contracts because the Rust config loader hands this list to the router as is:
whatever it changes in ids, ordering or name spelling changes which upstream a request reaches.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, gateway_from_environment, object_value
from tests.integration._support.wire import Wire
from tests.integration.configuration.dec1_cfgload_support import refused_with_exit, write_yaml_text
from tests.integration.configuration.dec1_config_support import (
    ITEMS,
    UPSTREAM_KEY,
    booted,
    chat,
    deployment,
    model_entries,
    scripted_upstream,
    sent,
    sent_json,
    served_model_names,
    write_config,
)

_RETRIES: Final = {"num_retries": 0}
_DECLARED_ID: Final = "dec1-cfgload-declared-id"


def test_a_declared_model_id_is_the_id_reported_on_every_boot(gateway: Gateway, tmp_path: Path) -> None:
    """One entry declares `model_info.id: dec1-cfgload-declared-id`. The proxy is booted twice from the
    same file, each time on its own scratch database. On both boots `GET /model/info` must report that
    id and a chat completion must answer with `x-litellm-model-id: dec1-cfgload-declared-id`: a declared
    id is used as written, never hashed or prefixed, so keys, spend rows and budgets that name it keep
    working across restarts and across a Python to Rust switch."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment("dec1-id", f"{upstream.url}/v1", model_info={"id": _DECLARED_ID})],
            "router_settings": _RETRIES,
        }
        observed: list[tuple[object, str | None]] = []
        for boot in ("first", "second"):
            with booted(gateway, tmp_path / boot, config) as owned:
                info_id = object_value(model_entries(owned.gateway, "dec1-id")[0]["model_info"])["id"]
                response = chat(owned.gateway, "dec1-id")
                assert response.status_code == 200, response.text
                observed.append((info_id, response.headers.get("x-litellm-model-id")))
    assert observed == [(_DECLARED_ID, _DECLARED_ID), (_DECLARED_ID, _DECLARED_ID)]


def test_two_entries_declaring_the_same_id_are_both_loaded_and_both_served(gateway: Gateway, tmp_path: Path) -> None:
    """Two entries with different names (`dec1-dup-a`, `dec1-dup-b`) and different upstreams both declare
    `model_info.id: dec1-cfgload-dup`. The proxy must boot, `GET /model/info` must list both names with
    that id, and a chat completion to each name must reach its own upstream: a repeated id is not a boot
    error outside PTU cost attribution (`_validated_deployment`, `ptu_identity_error`).
    UNVERIFIED: that `/model/info` lists both rather than collapsing them by id."""
    with scripted_upstream() as first, scripted_upstream() as second:
        config: Final = {
            "model_list": [
                deployment("dec1-dup-a", f"{first.url}/v1", model_info={"id": "dec1-cfgload-dup"}),
                deployment("dec1-dup-b", f"{second.url}/v1", model_info={"id": "dec1-cfgload-dup"}),
            ],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config) as owned:
            ids: Final = [
                object_value(entry["model_info"])["id"]
                for name in ("dec1-dup-a", "dec1-dup-b")
                for entry in model_entries(owned.gateway, name)
            ]
            assert chat(owned.gateway, "dec1-dup-a").status_code == 200
            assert chat(owned.gateway, "dec1-dup-b").status_code == 200
            reached: Final = (len(sent(first)), len(sent(second)))
    assert ids == ["dec1-cfgload-dup", "dec1-cfgload-dup"]
    assert reached == (1, 1)


def test_model_info_lookup_by_a_repeated_id_returns_the_later_entry(gateway: Gateway, tmp_path: Path) -> None:
    """Same two entries sharing `model_info.id: dec1-cfgload-dup`. `GET /model/info?litellm_model_id=
    dec1-cfgload-dup` must return exactly one row and it must be `dec1-dup-b`, the later entry: the id
    index keeps the last deployment pushed under an id (`registry.rs` `push_built`, `by_id.insert`).
    UNVERIFIED: that the Python `/model/info` id lookup goes through the same index."""
    config: Final = {
        "model_list": [
            deployment("dec1-dup-a", "http://127.0.0.1:9/v1", model_info={"id": "dec1-cfgload-dup"}),
            deployment("dec1-dup-b", "http://127.0.0.1:9/v1", model_info={"id": "dec1-cfgload-dup"}),
        ],
        "router_settings": _RETRIES,
    }
    with booted(gateway, tmp_path, config) as owned:
        rows: Final = ITEMS.validate_python(
            owned.gateway.get("/model/info", params={"litellm_model_id": "dec1-cfgload-dup"})["data"]
        )
    assert [row["model_name"] for row in rows] == ["dec1-dup-b"]


def test_an_organization_list_builds_one_deployment_per_organization(gateway: Gateway, tmp_path: Path) -> None:
    """One entry sets `litellm_params.organization: [org-dec1-a, org-dec1-b]`. `GET /model/info` must
    list two rows for the name, one with each organization, and a chat completion must reach the
    upstream with an `OpenAI-Organization` header naming one of the two: `set_model_list` adds one
    deployment per organization (`build.rs`, `organizations`).
    UNVERIFIED: the header name the OpenAI client sends (`openai-organization`)."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment("dec1-orgs", f"{upstream.url}/v1", organization=["org-dec1-a", "org-dec1-b"])],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config) as owned:
            organizations: Final = sorted(
                str(object_value(entry["litellm_params"])["organization"])
                for entry in model_entries(owned.gateway, "dec1-orgs")
            )
            assert chat(owned.gateway, "dec1-orgs").status_code == 200
            requests: Final = sent(upstream)
    assert organizations == ["org-dec1-a", "org-dec1-b"]
    assert [request.headers.get("openai-organization") in {"org-dec1-a", "org-dec1-b"} for request in requests] == [
        True
    ]


class _Wildcards:
    def __init__(self, catch_all: Wire, provider: Wire, family: Wire, exact: Wire, owned: Gateway) -> None:
        self.catch_all = catch_all
        self.provider = provider
        self.family = family
        self.exact = exact
        self.gateway = owned

    def reached(self) -> tuple[int, int, int, int]:
        return (len(sent(self.catch_all)), len(sent(self.provider)), len(sent(self.family)), len(sent(self.exact)))


@pytest.fixture(scope="module")
def wildcards(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Wildcards]:
    directory: Final = tmp_path_factory.mktemp("dec1-cfgload-wildcards")
    with (
        gateway_from_environment() as shared,
        scripted_upstream() as catch_all,
        scripted_upstream() as provider,
        scripted_upstream() as family,
        scripted_upstream() as exact,
    ):
        config: Final = {
            "model_list": [
                deployment("*", f"{catch_all.url}/v1", model="openai/*"),
                deployment("openai/*", f"{provider.url}/v1", model="openai/*"),
                deployment("openai/gpt-4o*", f"{family.url}/v1", model="openai/gpt-4o*"),
                deployment("openai/gpt-4o-dec1-exact", f"{exact.url}/v1", model="openai/gpt-4o-mini"),
            ],
            "router_settings": _RETRIES,
        }
        with booted(shared, directory, config) as owned:
            yield _Wildcards(catch_all, provider, family, exact, owned.gateway)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("openai/gpt-4o-dec1-exact", (0, 0, 0, 1)),
        ("openai/gpt-4o-dec1-other", (0, 0, 1, 0)),
        ("openai/o-dec1-model", (0, 1, 0, 0)),
        ("dec1-unprefixed-model", (1, 0, 0, 0)),
    ],
    ids=["exact_name", "longest_pattern", "provider_pattern", "catch_all"],
)
def test_the_most_specific_matching_name_serves_a_request(
    wildcards: _Wildcards, requested: str, expected: tuple[int, int, int, int]
) -> None:
    """One proxy has four entries: `*`, `openai/*`, `openai/gpt-4o*` and the exact
    `openai/gpt-4o-dec1-exact`, each on its own upstream. A request for the exact name must reach only
    the exact deployment, `openai/gpt-4o-dec1-other` only the `openai/gpt-4o*` one, `openai/o-dec1-model`
    only `openai/*`, and an unprefixed name only `*`: exact names beat patterns and a longer pattern
    beats a shorter one regardless of file order (`pattern_specificity`, `registry.rs:1137`)."""
    wildcards.reached()
    response: Final = chat(wildcards.gateway, requested)
    assert response.status_code == 200, response.text
    assert wildcards.reached() == expected


def test_a_wildcard_sends_the_requested_suffix_as_the_upstream_model(wildcards: _Wildcards) -> None:
    """Through the `openai/gpt-4o*` entry (whose `model` is also `openai/gpt-4o*`), a request for
    `openai/gpt-4o-dec1-other` must reach its upstream with body `model: gpt-4o-dec1-other`: the
    provider prefix is stripped and the matched name is sent, not the pattern."""
    sent(wildcards.family)
    assert chat(wildcards.gateway, "openai/gpt-4o-dec1-other").status_code == 200
    assert [sent_json(request)["model"] for request in sent(wildcards.family)] == ["gpt-4o-dec1-other"]


def test_a_wildcard_with_text_after_the_star_still_matches_a_longer_name(gateway: Gateway, tmp_path: Path) -> None:
    """The only entry is `dec1-*-mini` (model `openai/*`). A request for `dec1-team-mini-extended` must be
    served (200): a pattern is anchored at the start only, like `re.match` on `dec1-(.*)-mini`, so a name
    may run past the pattern's last literal (`wildcard_matches`, `registry.rs:1157`). A Rust matcher
    that anchors both ends would 400 here.
    UNVERIFIED: that the Python pattern router and the Rust matcher agree; the Rust doc comment says so."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment("dec1-*-mini", f"{upstream.url}/v1", model="openai/*")],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, "dec1-team-mini-extended")
    assert response.status_code == 200, response.text


def test_a_numeric_model_name_is_served_under_its_text(gateway: Gateway, tmp_path: Path) -> None:
    """The entry's `model_name` is the unquoted YAML integer `2026`. The proxy must boot and `GET
    /v1/models` must list the string `2026`, and a chat completion for `"2026"` must be served: the
    build stringifies the name (`_build_entry`, `str(model_name)`). A typed Rust `model_name: String`
    would refuse the integer (`litellm/rust/crates/config/src/model.rs`).
    UNVERIFIED: the Rust refusal is inferred from the struct type; serde_yaml_ng may not coerce."""
    with scripted_upstream() as upstream:
        text: Final = (
            "model_list:\n  - model_name: 2026\n    litellm_params:\n"
            f"      model: openai/gpt-4o-mini\n      api_base: {upstream.url}/v1\n      api_key: {UPSTREAM_KEY}\n"
            "router_settings:\n  num_retries: 0\n"
        )
        with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
            names: Final = served_model_names(owned.gateway)
            response: Final = chat(owned.gateway, "2026")
    assert "2026" in names
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "entry",
    [
        {"litellm_params": {"model": "openai/gpt-4o-mini", "api_key": UPSTREAM_KEY}},
        {"model_name": "dec1-no-params-mapping", "litellm_params": "openai/gpt-4o-mini"},
    ],
    ids=["missing_model_name", "litellm_params_not_a_mapping"],
)
def test_an_entry_without_a_name_or_with_scalar_params_refuses_to_boot(
    gateway: Gateway, tmp_path: Path, entry: dict[str, object]
) -> None:
    """A `model_list` entry has no `model_name`, or its `litellm_params` is a string instead of a
    mapping. The proxy must exit non-zero before becoming ready: a malformed entry raises where the
    build loop pops its fields (`_build_entry`).
    UNVERIFIED: the scalar `litellm_params` case is treated as `{}` by `_build_entry` and then fails
    for lack of `model`; the existing test of a missing `model` key covers the message."""
    config_path: Final = write_config(tmp_path, {"model_list": [entry], "router_settings": _RETRIES})
    refusal: Final = refused_with_exit(gateway, tmp_path, config_path)
    assert refusal.exit_code != 0


def test_a_model_list_that_is_a_mapping_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`model_list` is a mapping (`{dec1-model: {...}}`) instead of a list. The proxy must exit non-zero
    before becoming ready rather than serve no models, since a silently empty model list looks like an
    outage to callers.
    UNVERIFIED: the Python failure point; the Rust `Config` refuses it at deserialisation
    (`model_list: Box<[Model]>`)."""
    config_path: Final = write_config(
        tmp_path,
        {
            "model_list": {"dec1-model": {"litellm_params": {"model": "openai/gpt-4o-mini"}}},
            "router_settings": _RETRIES,
        },
    )
    refusal: Final = refused_with_exit(gateway, tmp_path, config_path)
    assert refusal.exit_code != 0


def test_an_unknown_key_on_a_model_entry_is_kept_out_of_the_provider_call(gateway: Gateway, tmp_path: Path) -> None:
    """A `model_list` entry carries an extra top-level key `dec1_note: hello` next to `model_name`,
    `litellm_params` and `model_info`. The proxy must boot, serve the model, and the upstream body must
    not contain `dec1_note`: unknown entry keys are tolerated (`Model.additional_fields` in Rust, `extras`
    in `_build_entry`) and never forwarded."""
    with scripted_upstream() as upstream:
        entry: Final = {**deployment("dec1-extra", f"{upstream.url}/v1"), "dec1_note": "hello"}
        with booted(gateway, tmp_path, {"model_list": [entry], "router_settings": _RETRIES}) as owned:
            assert chat(owned.gateway, "dec1-extra").status_code == 200
            requests: Final = sent(upstream)
    assert ["dec1_note" in sent_json(request) for request in requests] == [False]
