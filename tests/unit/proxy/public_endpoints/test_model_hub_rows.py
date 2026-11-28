from typing import Final

import pytest

import litellm
from litellm.proxy.public_endpoints.model_hub_rows import is_public_model_group, published_hub_rows

NAMED: Final = "hub-named"
HIDDEN: Final = "hub-hidden"
TEAM_INTERNAL: Final = "hub-team-internal"
TEAM_PUBLIC: Final = "hub-team-public"
TEAM_ID: Final = "hub-team"


def _router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {"model_name": NAMED, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "k"}},
            {
                "model_name": HIDDEN,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "k"},
                "model_info": {"discoverable": False},
            },
            {
                "model_name": TEAM_INTERNAL,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "k"},
                "model_info": {"team_id": TEAM_ID, "team_public_model_name": TEAM_PUBLIC},
            },
            {"model_name": "hubwild/*", "litellm_params": {"model": "openai/*", "api_key": "k"}},
        ]
    )


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> litellm.Router:
    serving: Final = _router()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", serving)
    return serving


async def _hub_names(monkeypatch: pytest.MonkeyPatch, public_model_groups: list[str]) -> tuple[str, ...]:
    monkeypatch.setattr(litellm, "public_model_groups", public_model_groups)
    rows: Final = await published_hub_rows()
    assert isinstance(rows, tuple), rows
    assert all(row.info.is_public_model_group for row in rows), rows
    return tuple(row.info.model_group for row in rows)


@pytest.mark.asyncio
@pytest.mark.usefixtures("router")
async def test_star_publishes_every_discoverable_group_and_wildcard_child(monkeypatch: pytest.MonkeyPatch):
    names: Final = await _hub_names(monkeypatch, ["*"])

    assert NAMED in names
    assert any(name.startswith("hubwild/") for name in names), names
    assert not any("*" in name for name in names), names
    assert HIDDEN not in names
    assert not {TEAM_INTERNAL, TEAM_PUBLIC} & set(names), names


@pytest.mark.asyncio
@pytest.mark.usefixtures("router")
async def test_a_name_listed_beside_star_is_published_even_when_star_leaves_it_out(monkeypatch: pytest.MonkeyPatch):
    names: Final = await _hub_names(monkeypatch, ["*", HIDDEN])

    assert {NAMED, HIDDEN} <= set(names), names


@pytest.mark.asyncio
@pytest.mark.usefixtures("router")
async def test_a_list_without_star_publishes_exactly_the_listed_names(monkeypatch: pytest.MonkeyPatch):
    assert await _hub_names(monkeypatch, [NAMED]) == (NAMED,)


@pytest.mark.parametrize(
    ("public_model_groups", "model_group", "team_id", "public"),
    [
        (["*"], NAMED, None, True),
        (["*"], "hubwild/gpt-4o", None, True),
        (["*"], HIDDEN, None, False),
        (["*"], TEAM_INTERNAL, None, False),
        (["*"], TEAM_PUBLIC, TEAM_ID, False),
        (["*"], NAMED, TEAM_ID, True),
        (["*", HIDDEN], HIDDEN, None, True),
        ([NAMED], NAMED, None, True),
        ([NAMED], "hubwild/gpt-4o", None, False),
        (None, NAMED, None, False),
    ],
)
def test_is_public_model_group(
    monkeypatch: pytest.MonkeyPatch,
    public_model_groups: list[str] | None,
    model_group: str,
    team_id: str | None,
    public: bool,
):
    monkeypatch.setattr(litellm, "public_model_groups", public_model_groups)

    assert is_public_model_group(model_group, _router(), team_id) is public
