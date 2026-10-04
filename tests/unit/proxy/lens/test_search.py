import pytest

from litellm.proxy.lens.search import LensFilter, LensSearch, parse_search


@pytest.mark.parametrize(
    "q, text, filters",
    (
        pytest.param("", (), (), id="empty"),
        pytest.param("foo  bar", ("%foo%", "%bar%"), (), id="words"),
        pytest.param('"foo bar"', ("%foo bar%",), (), id="quoted-phrase"),
        pytest.param('"foo bar', ("%foo bar%",), (), id="unclosed-quote"),
        pytest.param("50%_off\\", ("%50\\%\\_off\\\\%",), (), id="like-metacharacters"),
        pytest.param('name:"nightly scan"', (), (LensFilter("name", "nightly scan", False),), id="exact"),
        pytest.param("agent:res*er", (), (LensFilter("agent", "res%er", False),), id="glob"),
        pytest.param("name:a_b*", (), (LensFilter("name", "a\\_b%", False),), id="glob-escapes-rest"),
        pytest.param("-schedule:paused", (), (LensFilter("schedule", "paused", True),), id="negated"),
        pytest.param("STATUS:failed", (), (LensFilter("status", "failed", False),), id="key-ignores-case"),
        pytest.param("name:a:b", (), (LensFilter("name", "a:b", False),), id="value-keeps-colons"),
        pytest.param("status: foo", ("%foo%",), (), id="missing-value-narrows-nothing"),
        pytest.param("model:x", ("%model:x%",), (), id="unknown-key-is-text"),
        pytest.param("-foo", ("%-foo%",), (), id="negated-text-stays-text"),
    ),
)
def test_parse_matches_the_dashboard_search_grammar(
    q: str, text: tuple[str, ...], filters: tuple[LensFilter, ...]
) -> None:
    assert parse_search(q) == LensSearch(text=text, filters=filters)
