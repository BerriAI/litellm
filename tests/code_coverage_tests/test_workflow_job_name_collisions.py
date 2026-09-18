from typing import Final

from check_workflow_job_name_collisions import (
    Unreadable,
    blind_spots,
    callee_path,
    collisions,
    exit_code,
    parse,
    published,
    unreadable,
    workflow_sources,
)

REUSABLE_BASE: Final = """on:
  workflow_call:
jobs:
  run:
    name: >-
      ${{ matrix.python-version == '3.12' && 'Run tests'
      || format('Run tests (Python {0})', matrix.python-version) }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
"""

SHARD_CALLER: Final = """on: pull_request
jobs:
  unit:
    name: ${{ matrix.shard }}
    uses: ./.github/workflows/base.yml
    strategy:
      matrix:
        include:
          - shard: core-utils
"""


CORRELATED_ROWS: Final = """on: pull_request
jobs:
  unit:
    name: ${{ matrix.shard }} on ${{ matrix.test-path }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - shard: core-utils
            test-path: tests/core
          - shard: proxy
            test-path: tests/proxy
"""

LISTED_PLUS_ROW: Final = """on: pull_request
jobs:
  unit:
    name: ${{ matrix.python-version }} ${{ matrix.label }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
        include:
          - label: fast
"""

NAMELESS_MATRIX: Final = """on: pull_request
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
"""

EXCLUDED_PAIR: Final = """on: pull_request
jobs:
  unit:
    name: ${{ matrix.os }}-${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu, macos]
        python-version: ["3.12", "3.13"]
        exclude:
          - os: macos
            python-version: "3.13"
"""

EXCLUDED_KEY: Final = """on: pull_request
jobs:
  unit:
    name: ${{ matrix.os }}-${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu, macos]
        python-version: ["3.12", "3.13"]
        exclude:
          - os: macos
"""

BOOLEAN_MATRIX: Final = """on: pull_request
jobs:
  unit:
    name: cache ${{ matrix.cached }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        cached: [true, false]
"""

UNFILLABLE_FORMAT: Final = """on: pull_request
jobs:
  unit:
    name: ${{ format('{0} {1}', matrix.shard) }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [core]
"""


def test_every_workflow_in_the_repo_publishes_a_unique_check_run_name() -> None:
    assert collisions(workflow_sources()) == ()


def test_every_workflow_in_the_repo_parses_into_jobs() -> None:
    unparsed: Final = tuple(
        rel
        for rel, source in workflow_sources().items()
        if isinstance(entry := parse(source), Unreadable) or not entry[0].jobs
    )

    assert unparsed == ()


def test_every_local_reusable_call_in_the_repo_resolves_to_a_workflow() -> None:
    sources: Final = workflow_sources()
    parsed: Final = tuple(entry for text in sources.values() if not isinstance(entry := parse(text), Unreadable))
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


def test_two_workflows_sharing_a_run_wide_template_are_not_called_a_collision() -> None:
    template: Final = (
        "on: pull_request\njobs:\n  {job}:\n    name: ${{{{ github.event_name }}}}-build\n    runs-on: ubuntu-latest\n"
    )
    sources: Final = {
        "a.yml": template.format(job="one"),
        "b.yml": template.format(job="two"),
    }

    assert collisions(sources) == ()
    assert blind_spots(sources) == ()


def test_two_jobs_of_one_workflow_sharing_a_run_wide_template_are_a_collision() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n"
            "  one:\n    name: ${{ github.event_name }}-build\n    runs-on: ubuntu-latest\n"
            "  two:\n    name: ${{ github.event_name }}-build\n    runs-on: ubuntu-latest\n"
        ),
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "is published 2 times inside a.yml, by job `one`, job `two`" in found[0]
    assert exit_code(sources) == 1


def test_a_run_wide_template_carrying_a_matrix_value_does_not_collide_inside_one_workflow() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n"
            "  one:\n    name: ${{ github.event_name }}-${{ matrix.shard }}\n    runs-on: ubuntu-latest\n"
            "    strategy:\n      matrix:\n        shard: [core, extras]\n"
        ),
    }

    assert collisions(sources) == ()
    assert blind_spots(sources) == ()


def test_a_run_wide_name_repeated_over_a_matrix_by_one_job_is_a_collision() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n"
            "  one:\n    name: ${{ github.event_name }}-build\n    runs-on: ubuntu-latest\n"
            "    strategy:\n      matrix:\n        shard: [core, extras]\n"
        ),
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "is published 2 times inside a.yml, by job `one`" in found[0]


def test_a_name_reading_the_job_it_sits_in_stays_out_of_the_comparison() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n"
            "  one:\n    name: ${{ github.job }}-build\n    runs-on: ubuntu-latest\n"
            "  two:\n    name: ${{ github.job }}-build\n    runs-on: ubuntu-latest\n"
        ),
    }

    assert collisions(sources) == ()
    assert len(blind_spots(sources)) == 2


def test_a_run_wide_caller_name_collides_through_the_workflow_it_calls() -> None:
    sources: Final = {
        ".github/workflows/a.yml": (
            "on: pull_request\njobs:\n"
            "  one:\n    name: ${{ github.event_name }}\n    uses: ./.github/workflows/c.yml\n"
            "  two:\n    name: ${{ github.event_name }}\n    uses: ./.github/workflows/c.yml\n"
        ),
        ".github/workflows/c.yml": "on:\n  workflow_call:\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "github.event_name }} / build` is published 2 times inside .github/workflows/a.yml" in found[0]


def test_a_run_wide_name_inside_a_called_workflow_collides_under_the_caller() -> None:
    sources: Final = {
        ".github/workflows/a.yml": ("on: pull_request\njobs:\n  one:\n    uses: ./.github/workflows/c.yml\n"),
        ".github/workflows/c.yml": (
            "on:\n  workflow_call:\njobs:\n"
            "  build:\n    name: ${{ github.event_name }}\n    runs-on: ubuntu-latest\n"
            "  lint:\n    name: ${{ github.event_name }}\n    runs-on: ubuntu-latest\n"
        ),
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`one / ${{ github.event_name }}` is published 2 times inside .github/workflows/a.yml" in found[0]


def test_a_name_reading_the_workflow_it_sits_in_is_not_called_a_collision() -> None:
    template: Final = (
        "on: pull_request\njobs:\n  {job}:\n    name: ${{{{ github.workflow }}}} / build\n    runs-on: ubuntu-latest\n"
    )
    sources: Final = {
        "a.yml": template.format(job="one"),
        "b.yml": template.format(job="two"),
    }

    assert collisions(sources) == ()
    assert exit_code(sources) == 0


def test_a_format_call_python_accepts_but_github_does_not_publishes_nothing_to_compare() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  one:\n    name: ${{ format('{0.real}', matrix.shard) }}\n"
            "    runs-on: ubuntu-latest\n    strategy:\n      matrix:\n        shard: [core]\n"
        ),
    }

    assert collisions(sources) == ()
    assert blind_spots(sources) != ()


def test_a_format_call_padding_its_argument_publishes_nothing_to_compare() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  one:\n    name: ${{ format('{0:>8}', matrix.shard) }}\n"
            "    runs-on: ubuntu-latest\n    strategy:\n      matrix:\n        shard: [core]\n"
        ),
        "b.yml": "on: pull_request\njobs:\n  two:\n    name: '    core'\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()
    assert blind_spots(sources) != ()


def test_an_exclude_row_that_is_not_a_mapping_is_reported_rather_than_skipped() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
            "    strategy:\n      matrix:\n        v: [1, 2]\n        exclude:\n          - oops\n"
        ),
        "b.yml": "on: pull_request\njobs:\n  other:\n    name: build (1)\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()
    assert blind_spots(sources) != ()


def test_an_exclude_row_holding_a_non_scalar_never_drops_every_combination() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
            "    strategy:\n      matrix:\n        v: [1, 2]\n        exclude:\n          - cfg: {k: 1}\n"
        ),
        "b.yml": "on: pull_request\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()
    assert blind_spots(sources) != ()


def test_two_jobs_sharing_a_template_that_reads_per_job_are_not_called_a_collision() -> None:
    template: Final = (
        "on: pull_request\njobs:\n  {job}:\n    name: ${{{{ matrix.shard }}}}\n    runs-on: ubuntu-latest\n"
    )
    sources: Final = {
        "a.yml": template.format(job="one"),
        "b.yml": template.format(job="two"),
    }

    assert collisions(sources) == ()
    assert len(blind_spots(sources)) == 2


def test_a_file_that_is_not_a_workflow_is_reported_rather_than_skipped() -> None:
    sources: Final = {
        "notes.yml": "just a string\n",
        "a.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
    }

    found: Final = unreadable(sources)

    assert len(found) == 1
    assert "notes.yml" in found[0]
    assert collisions(sources) == ()


def test_a_workflow_holding_a_job_shape_github_would_reject_is_reported() -> None:
    sources: Final = {"a.yml": "on: pull_request\njobs:\n  test:\n    uses: [not, a, string]\n"}

    found: Final = unreadable(sources)

    assert len(found) == 1
    assert "a.yml" in found[0]


def test_a_conditional_name_expands_to_the_branch_each_matrix_value_takes() -> None:
    names: Final = frozenset(
        name for name, _ in published({".github/workflows/base.yml": REUSABLE_BASE, "unit.yml": SHARD_CALLER})
    )

    assert names == frozenset({"core-utils / Run tests", "core-utils / Run tests (Python 3.13)"})


def test_a_conditional_name_never_publishes_the_branch_its_condition_rules_out() -> None:
    names: Final = frozenset(
        name for name, _ in published({".github/workflows/base.yml": REUSABLE_BASE, "unit.yml": SHARD_CALLER})
    )

    assert "core-utils / Run tests (Python 3.12)" not in names


def test_a_conditional_reusable_name_collides_with_a_plain_job_publishing_the_same_name() -> None:
    sources: Final = {
        ".github/workflows/base.yml": REUSABLE_BASE,
        "unit.yml": SHARD_CALLER,
        "postgres.yml": ("on: pull_request\njobs:\n  legacy:\n    name: core-utils / Run tests\n"),
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`core-utils / Run tests` is published by 2 jobs" in found[0]


def test_a_name_reading_two_matrix_keys_publishes_only_the_pairs_each_include_row_holds() -> None:
    names: Final = frozenset(name for name, _ in published({"unit.yml": CORRELATED_ROWS}))

    assert names == frozenset({"core-utils on tests/core", "proxy on tests/proxy"})


def test_a_name_reading_two_matrix_keys_never_publishes_a_pair_no_include_row_holds() -> None:
    names: Final = frozenset(name for name, _ in published({"unit.yml": CORRELATED_ROWS}))

    assert "core-utils on tests/proxy" not in names
    assert "proxy on tests/core" not in names


def test_an_include_row_carrying_no_listed_key_extends_every_listed_combination() -> None:
    names: Final = frozenset(name for name, _ in published({"unit.yml": LISTED_PLUS_ROW}))

    assert names == frozenset({"3.12 fast", "3.13 fast"})


def test_every_workflow_in_the_repo_resolves_every_expression_in_its_job_names() -> None:
    unresolved: Final = tuple(f"{owner}: {name}" for name, owner in published(workflow_sources()) if "${{" in name)

    assert unresolved == ()


def test_a_matrix_job_with_no_name_publishes_the_id_and_values_github_appends() -> None:
    names: Final = frozenset(name for name, _ in published({"a.yml": NAMELESS_MATRIX}))

    assert names == frozenset({"build (3.12)", "build (3.13)"})


def test_a_matrix_job_with_no_name_does_not_collide_with_a_plain_job_carrying_its_id() -> None:
    sources: Final = {
        "a.yml": NAMELESS_MATRIX,
        "b.yml": "on: pull_request\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
    }

    assert collisions(sources) == ()


def test_a_matrix_job_with_no_name_collides_with_the_suffixed_name_github_writes() -> None:
    sources: Final = {
        "a.yml": NAMELESS_MATRIX,
        "b.yml": "on: pull_request\njobs:\n  legacy:\n    name: build (3.13)\n    runs-on: ubuntu-latest\n",
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`build (3.13)` is published by 2 jobs" in found[0]


def test_an_excluded_combination_publishes_no_check_run() -> None:
    names: Final = frozenset(name for name, _ in published({"unit.yml": EXCLUDED_PAIR}))

    assert names == frozenset({"ubuntu-3.12", "ubuntu-3.13", "macos-3.12"})


def test_an_exclude_row_naming_one_key_drops_every_combination_carrying_it() -> None:
    names: Final = frozenset(name for name, _ in published({"unit.yml": EXCLUDED_KEY}))

    assert names == frozenset({"ubuntu-3.12", "ubuntu-3.13"})


def test_a_boolean_matrix_value_renders_the_way_github_writes_it() -> None:
    names: Final = frozenset(name for name, _ in published({"unit.yml": BOOLEAN_MATRIX}))

    assert names == frozenset({"cache true", "cache false"})


def test_a_format_call_its_arguments_cannot_fill_publishes_nothing_to_compare() -> None:
    sources: Final = {"unit.yml": UNFILLABLE_FORMAT}

    assert frozenset(name for name, _ in published(sources)) == frozenset()
    assert "its name stays" in blind_spots(sources)[0]


def test_a_call_to_a_workflow_outside_the_repo_is_reported_rather_than_guessed() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  unit:\n    uses: BerriAI/other/.github/workflows/base.yml@main\n",
        "b.yml": "on: pull_request\njobs:\n  unit:\n    runs-on: ubuntu-latest\n",
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"unit"})
    assert collisions(sources) == ()
    assert "outside this repository" in blind_spots(sources)[0]


def test_a_chain_of_local_reusable_calls_publishes_every_level_of_the_chain() -> None:
    sources: Final = {
        ".github/workflows/leaf.yml": (
            "on:\n  workflow_call:\njobs:\n  run:\n    name: Leaf\n    runs-on: ubuntu-latest\n"
        ),
        ".github/workflows/mid.yml": (
            "on:\n  workflow_call:\njobs:\n  call:\n    name: Mid\n    uses: ./.github/workflows/leaf.yml\n"
        ),
        "top.yml": "on: pull_request\njobs:\n  top:\n    name: Top\n    uses: ./.github/workflows/mid.yml\n",
    }

    names: Final = frozenset(name for name, _ in published(sources))

    assert names == frozenset({"Top / Mid / Leaf"})


def test_a_job_name_that_is_not_a_string_still_publishes_the_value_github_renders() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  sweep:\n    name: 2024\n    runs-on: ubuntu-latest\n",
        "b.yml": 'on: pull_request\njobs:\n  other:\n    name: "2024"\n    runs-on: ubuntu-latest\n',
    }

    found: Final = collisions(sources)

    assert len(found) == 1
    assert "`2024` is published by 2 jobs" in found[0]


def test_the_check_fails_when_a_file_in_the_workflows_directory_cannot_be_read() -> None:
    assert exit_code({"notes.yml": "just a string\n"}) == 1


def test_the_check_fails_when_two_jobs_publish_one_check_run_name() -> None:
    plain: Final = "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n"

    assert exit_code({"a.yml": plain, "b.yml": plain}) == 1


def test_the_check_passes_when_every_file_reads_and_every_name_is_unique() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
        "b.yml": "on: pull_request\njobs:\n  sweep:\n    runs-on: ubuntu-latest\n",
    }

    assert exit_code(sources) == 0


def test_two_callers_of_one_reusable_workflow_named_from_its_inputs_do_not_collide() -> None:
    sources: Final = {
        ".github/workflows/callee.yml": (
            "on:\n  workflow_call:\njobs:\n  run:\n    name: ${{ inputs.suite }}\n    runs-on: ubuntu-latest\n"
        ),
        "caller.yml": (
            "on: pull_request\njobs:\n"
            "  alpha:\n    name: A\n    uses: ./.github/workflows/callee.yml\n    with:\n      suite: alpha\n"
            "  beta:\n    name: A\n    uses: ./.github/workflows/callee.yml\n    with:\n      suite: beta\n"
        ),
    }

    assert collisions(sources) == ()
    assert len(blind_spots(sources)) == 2


def test_a_matrix_that_is_itself_an_expression_never_collapses_onto_the_bare_job_id() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  build:\n    strategy:\n"
            "      matrix: ${{ fromJson(needs.plan.outputs.matrix) }}\n    runs-on: ubuntu-latest\n"
        ),
        "b.yml": "on: pull_request\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"build"})
    assert collisions(sources) == ()
    assert "the matrix itself comes from an expression" in blind_spots(sources)[0]


def test_a_matrix_listing_objects_never_collapses_onto_the_bare_job_id() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  build:\n    strategy:\n      matrix:\n        target:\n"
            "          - os: ubuntu\n          - os: windows\n    runs-on: ubuntu-latest\n"
        ),
        "b.yml": "on: pull_request\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"build"})
    assert collisions(sources) == ()
    assert "not plain scalars" in blind_spots(sources)[0]


def test_a_call_to_a_workflow_file_the_checkout_does_not_hold_is_reported() -> None:
    sources: Final = {"a.yml": "on: pull_request\njobs:\n  unit:\n    uses: ./.github/workflows/gone.yml\n"}

    assert collisions(sources) == ()
    assert "which this checkout does not hold" in blind_spots(sources)[0]


def test_reusable_workflows_calling_each_other_in_a_loop_are_reported_not_followed() -> None:
    sources: Final = {
        ".github/workflows/a.yml": (
            "on:\n  workflow_call:\njobs:\n  call:\n    name: A\n    uses: ./.github/workflows/b.yml\n"
        ),
        ".github/workflows/b.yml": (
            "on:\n  workflow_call:\njobs:\n  call:\n    name: B\n    uses: ./.github/workflows/a.yml\n"
        ),
        "top.yml": "on: pull_request\njobs:\n  top:\n    name: Top\n    uses: ./.github/workflows/a.yml\n",
    }

    assert collisions(sources) == ()
    assert any("loops back on itself" in spot for spot in blind_spots(sources))


def test_a_caller_still_publishes_the_callee_jobs_it_can_read() -> None:
    sources: Final = {
        ".github/workflows/callee.yml": (
            "on:\n  workflow_call:\njobs:\n"
            "  lint:\n    name: Lint\n    runs-on: ubuntu-latest\n"
            "  suite:\n    name: ${{ inputs.suite }}\n    runs-on: ubuntu-latest\n"
        ),
        "caller.yml": "on: pull_request\njobs:\n  call:\n    name: A\n    uses: ./.github/workflows/callee.yml\n",
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"A / Lint"})
    assert len(blind_spots(sources)) == 1


def test_a_name_the_check_cannot_work_out_is_reported_without_failing_the_check() -> None:
    sources: Final = {
        "a.yml": "on: pull_request\njobs:\n  unit:\n    uses: BerriAI/other/.github/workflows/base.yml@main\n",
    }

    assert blind_spots(sources) != ()
    assert exit_code(sources) == 0


def test_a_caller_whose_own_name_is_unreadable_publishes_none_of_its_callee_names() -> None:
    sources: Final = {
        ".github/workflows/callee.yml": (
            "on:\n  workflow_call:\njobs:\n  lint:\n    name: Lint\n    runs-on: ubuntu-latest\n"
        ),
        "caller.yml": (
            "on: pull_request\njobs:\n  call:\n    name: ${{ matrix.suite }}\n"
            "    uses: ./.github/workflows/callee.yml\n"
        ),
        "other.yml": "on: pull_request\njobs:\n  plain:\n    name: Lint\n    runs-on: ubuntu-latest\n",
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"Lint"})
    assert collisions(sources) == ()
    assert "its name stays" in blind_spots(sources)[0]


def test_an_include_row_naming_a_listed_key_extends_only_the_combinations_it_matches() -> None:
    sources: Final = {
        "unit.yml": (
            "on: pull_request\njobs:\n  unit:\n"
            "    name: ${{ matrix.python-version }} ${{ matrix.label }}\n"
            "    runs-on: ubuntu-latest\n    strategy:\n      matrix:\n"
            '        python-version: ["3.12", "3.13"]\n'
            "        include:\n"
            '          - python-version: "3.12"\n'
            "            label: fast\n"
        )
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"3.12 fast"})
    assert len(blind_spots(sources)) == 1


def test_a_job_whose_whole_strategy_is_an_expression_is_reported_rather_than_rejecting_the_file() -> None:
    sources: Final = {
        "plan.yml": (
            "on: pull_request\njobs:\n  plan:\n    name: Plan\n    runs-on: ubuntu-latest\n"
            "  fan:\n    strategy: ${{ fromJSON(needs.plan.outputs.strategy) }}\n    runs-on: ubuntu-latest\n"
        )
    }

    assert unreadable(sources) == ()
    assert frozenset(name for name, _ in published(sources)) == frozenset({"Plan"})
    assert "`strategy` comes from an expression" in blind_spots(sources)[0]
    assert exit_code(sources) == 0


def test_one_job_publishing_one_name_for_every_matrix_combination_is_a_collision() -> None:
    sources: Final = {
        "unit.yml": (
            "on: pull_request\njobs:\n  build:\n    name: Run tests\n    runs-on: ubuntu-latest\n"
            '    strategy:\n      matrix:\n        python-version: ["3.12", "3.13"]\n'
        )
    }

    found: Final = collisions(sources)
    assert len(found) == 1
    assert "`Run tests` is published 2 times by unit.yml job `build`" in found[0]
    assert exit_code(sources) == 1


def test_a_name_carrying_a_matrix_value_publishes_one_name_per_combination_without_colliding() -> None:
    sources: Final = {
        "unit.yml": (
            "on: pull_request\njobs:\n  build:\n    name: Run tests ${{ matrix.python-version }}\n"
            '    runs-on: ubuntu-latest\n    strategy:\n      matrix:\n        python-version: ["3.12", "3.13"]\n'
        )
    }

    assert frozenset(name for name, _ in published(sources)) == frozenset({"Run tests 3.12", "Run tests 3.13"})
    assert collisions(sources) == ()
    assert exit_code(sources) == 0


def test_a_file_that_is_not_valid_yaml_is_reported_rather_than_raising() -> None:
    sources: Final = {"broken.yml": "jobs:\n  build: [\n"}

    assert unreadable(sources) == (
        "broken.yml sits in the workflows directory but it does not read as one YAML "
        "document, so none of its jobs were checked.",
    )
    assert exit_code(sources) == 1


def test_a_file_holding_two_yaml_documents_is_reported_rather_than_raising() -> None:
    sources: Final = {"two.yml": "on: pull_request\n---\non: push\n"}

    assert len(unreadable(sources)) == 1
    assert exit_code(sources) == 1


def test_an_exclude_that_is_itself_an_expression_is_reported_rather_than_ignored() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  one:\n    name: build\n    runs-on: ubuntu-latest\n"
            "    strategy:\n      matrix:\n        python: ['3.11', '3.12']\n"
            "        exclude: ${{ fromJson(vars.SKIP) }}\n"
        ),
    }

    found: Final = blind_spots(sources)

    assert collisions(sources) == ()
    assert len(found) == 1
    assert "a matrix `exclude` is itself an expression" in found[0]


def test_an_include_that_is_itself_an_expression_is_reported_rather_than_ignored() -> None:
    sources: Final = {
        "a.yml": (
            "on: pull_request\njobs:\n  one:\n    name: build-${{ matrix.python }}\n    runs-on: ubuntu-latest\n"
            "    strategy:\n      matrix:\n        python: ['3.11']\n"
            "        include: ${{ fromJson(vars.EXTRA) }}\n"
        ),
    }

    found: Final = blind_spots(sources)

    assert collisions(sources) == ()
    assert len(found) == 1
    assert "a matrix `include` is itself an expression" in found[0]
