from scripts.generate_owned_kwargs import TARGET, render


def test_rust_owned_kwargs_match_the_python_owned_set() -> None:
    assert TARGET.read_text() == render(), "stale: run `python -m scripts.generate_owned_kwargs`"
