from scripts import generate_owned_kwargs


def test_owned_rust_copy_matches_the_python_set() -> None:
    assert generate_owned_kwargs.TARGET.read_text() == generate_owned_kwargs.render()
