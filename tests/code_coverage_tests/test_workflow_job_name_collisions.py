import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_workflow_job_name_collisions import callee_path, collisions, parse, workflow_sources  # noqa: E402


def test_every_workflow_in_the_repo_publishes_a_unique_check_run_name() -> None:
    assert collisions(workflow_sources()) == ()


def test_every_workflow_in_the_repo_parses_into_jobs() -> None:
    unparsed: Final = tuple(
        rel for rel, source in workflow_sources().items() if (entry := parse(source)) is None or not entry[0].jobs
    )

    assert unparsed == ()


def test_every_local_reusable_call_in_the_repo_resolves_to_a_workflow() -> None:
    sources: Final = workflow_sources()
    parsed: Final = tuple(entry for text in sources.values() if (entry := parse(text)) is not None)
    unresolved: Final = tuple(
        job.uses
        for workflow, _ in parsed
        for job in workflow.jobs.values()
        if (callee := callee_path(job)) is not None and callee not in sources
    )

    assert unresolved == ()


def test_two_jobs_falling_back_to_the_same_job_id_collide() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
        "b.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`test` is published by 2 jobs" in found[0]
    assert "a.yml job `test`" in found[0] and "b.yml job `test`" in found[0]


def test_an_explicit_name_overrides_the_job_id_and_clears_the_collision() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  test:\n    name: Sweep tests\n    runs-on: ubuntu-latest\n",
        "b.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()


def test_an_explicit_name_matching_another_job_id_collides() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  sweep:\n    name: test\n    runs-on: ubuntu-latest\n",
        "b.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`test` is published by 2 jobs" in found[0]


def test_two_callers_of_one_reusable_workflow_collide_on_a_shared_matrix_value() -> None:
    base: Final = "on:\n  workflow_call:\njobs:\n  run:\n    name: Run tests\n    runs-on: ubuntu-latest\n"
    caller: Final = (
        "on: pull_request\n"
        "jobs:\n"
        "  {job}:\n"
        "    name: ${{{{ matrix.shard }}}}\n"
        "    uses: ./.github/workflows/base.yml\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        "          - shard: {shard}\n"
    )
    sources: Final = {
        ".github/workflows/base.yml": base,
        "unit.yml": caller.format(job="unit", shard="proxy-auth"),
        "proxy-db.yml": caller.format(job="proxy-db", shard="proxy-auth"),
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`proxy-auth / Run tests` is published by 2 jobs" in found[0]


def test_distinct_matrix_values_through_one_reusable_workflow_do_not_collide() -> None:
    base: Final = "on:\n  workflow_call:\njobs:\n  run:\n    name: Run tests\n    runs-on: ubuntu-latest\n"
    caller: Final = (
        "on: pull_request\n"
        "jobs:\n"
        "  {job}:\n"
        "    name: ${{{{ matrix.shard }}}}\n"
        "    uses: ./.github/workflows/base.yml\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        "          - shard: {shard}\n"
    )
    sources: Final = {
        ".github/workflows/base.yml": base,
        "unit.yml": caller.format(job="unit", shard="proxy-auth"),
        "proxy-db.yml": caller.format(job="proxy-db", shard="budgets"),
    }

    assert collisions(sources) == ()


def test_a_reusable_caller_does_not_collide_with_a_plain_job_of_the_same_name() -> None:
    sources: Final = {
        ".github/workflows/base.yml": (
            "on:\n  workflow_call:\njobs:\n  run:\n    name: Run tests\n    runs-on: ubuntu-latest\n"
        ),
        "unit.yml": (
            "on: pull_request\n"
            "jobs:\n"
            "  unit:\n"
            "    name: ${{ matrix.shard }}\n"
            "    uses: ./.github/workflows/base.yml\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - shard: proxy-behavior\n"
        ),
        "postgres.yml": (
            "on: pull_request\n"
            "jobs:\n"
            "  postgres:\n"
            "    name: ${{ matrix.shard }}\n"
            "    runs-on: ubuntu-latest\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - shard: proxy-behavior\n"
        ),
    }

    assert collisions(sources) == ()


def test_a_workflow_call_only_workflow_publishes_nothing_of_its_own() -> None:
    sources: Final = {
        "base.yml": "on:\n  workflow_call:\njobs:\n  run:\n    runs-on: ubuntu-latest\n",
        "other.yml": "on:\n  workflow_call:\njobs:\n  run:\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()


def test_a_workflow_call_workflow_that_also_runs_on_pull_request_still_publishes() -> None:
    sources: Final = {
        "base.yml": "on:\n  workflow_call:\n  pull_request:\njobs:\n  run:\n    runs-on: ubuntu-latest\n",
        "other.yml": "on: pull_request\njobs:\n  run:\n    runs-on: ubuntu-latest\n",
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`run` is published by 2 jobs" in found[0]


def test_a_matrix_list_supplies_values_the_same_way_include_rows_do() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\n"
            "jobs:\n"
            "  build:\n"
            "    name: Analyze (${{ matrix.language }})\n"
            "    runs-on: ubuntu-latest\n"
            "    strategy:\n"
            "      matrix:\n"
            "        language: [python, go]\n"
        ),
        "b.yml": "on: pull_request\njobs:\n  go:\n    name: Analyze (go)\n    runs-on: ubuntu-latest\n",
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`Analyze (go)` is published by 2 jobs" in found[0]


def test_two_jobs_sharing_one_unresolvable_template_still_collide() -> None:
    template: Final = (
        "on: pull_request\njobs:\n  {job}:\n    name: ${{{{ matrix.shard }}}}\n    runs-on: ubuntu-latest\n"
    )
    sources: Final = {
        "a.yml": template.format(job="one"),
        "b.yml": template.format(job="two"),
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "is published by 2 jobs" in found[0]


def test_a_file_that_is_not_a_workflow_is_ignored() -> None:
    sources: Final = {
        "notes.yml": "just a string\n",
        "a.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()
