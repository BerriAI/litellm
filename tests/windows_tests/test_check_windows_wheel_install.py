import zipfile

import pytest

from check_windows_wheel_install import (
    MAX_DIRECTORY_PATH,
    MAX_PATH,
    WORST_CASE_PREFIX,
    main,
    overlong_install_paths,
)

FILE_BUDGET = MAX_PATH - WORST_CASE_PREFIX - 1
DIRECTORY_BUDGET = MAX_DIRECTORY_PATH - WORST_CASE_PREFIX - 1


def _wheel(tmp_path, *entry_names):
    path = tmp_path / "pkg.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name in entry_names:
            zf.writestr(name, "{}")
    return str(path)


def test_flags_entry_one_char_over_budget(tmp_path):
    busts = "a" * (FILE_BUDGET + 1)
    assert overlong_install_paths(_wheel(tmp_path, busts)) == [busts]


def test_allows_entry_exactly_at_budget(tmp_path):
    at_limit = "a" * FILE_BUDGET
    assert (
        overlong_install_paths(_wheel(tmp_path, at_limit, "litellm/__init__.py")) == []
    )


def test_flags_directory_one_char_over_create_directory_limit(tmp_path):
    busts = "d" * (DIRECTORY_BUDGET + 1) + "/f"
    assert overlong_install_paths(_wheel(tmp_path, busts)) == [busts]


def test_allows_directory_exactly_at_create_directory_limit(tmp_path):
    at_limit = "d" * DIRECTORY_BUDGET + "/f"
    assert overlong_install_paths(_wheel(tmp_path, at_limit)) == []


@pytest.mark.parametrize(
    "entry",
    [
        "litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/guardrail_benchmarks/evals/block_disability_discrimination.jsonl",
        "litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/policy_templates/sg_pdpa_profiling_automated_decisions.yaml",
    ],
)
def test_flags_the_paths_that_overflowed_store_python(tmp_path, entry):
    """Both shipped in v1.103.1 and broke pip install under Microsoft Store Python (#43851)."""
    assert overlong_install_paths(_wheel(tmp_path, entry)) == [entry]


def test_orders_offenders_longest_first(tmp_path):
    longer = "a" * (FILE_BUDGET + 5)
    shorter = "b" * (FILE_BUDGET + 1)
    assert overlong_install_paths(_wheel(tmp_path, shorter, longer)) == [
        longer,
        shorter,
    ]


def _dist_with(tmp_path, *entry_names):
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "litellm-0-py3-none-any.whl", "w") as zf:
        for name in entry_names:
            zf.writestr(name, "{}")


def test_lengths_only_passes_without_installing(tmp_path, monkeypatch):
    _dist_with(tmp_path, "litellm/__init__.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    assert main(["--lengths-only"]) == 0


def test_lengths_only_fails_on_an_overlong_path(tmp_path, monkeypatch):
    _dist_with(tmp_path, "a" * (FILE_BUDGET + 1))
    monkeypatch.chdir(tmp_path)
    assert main(["--lengths-only"]) == 1
