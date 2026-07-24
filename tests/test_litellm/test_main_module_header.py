from pathlib import Path


def test_main_py_starts_with_brief_file_description():
    repo_root = Path(__file__).resolve().parents[2]
    main_py = repo_root / "litellm" / "main.py"

    first_two_lines = main_py.read_text(encoding="utf-8").splitlines()[:2]

    assert any(
        "LiteLLM main module" in line and "entrypoints" in line
        for line in first_two_lines
    )
