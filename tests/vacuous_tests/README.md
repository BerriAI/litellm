# Vacuous test audit

A vacuous test is one that cannot fail when the code it claims to test is broken. It passes CI, adds runtime, and gives false confidence, which is worse than having no test at all

This directory holds the tooling behind two things: a CI ratchet that stops new vacuous tests from landing, and a daily automation that fixes existing ones

## Stage A: candidate inventory

`inventory.py` walks `tests/` and classifies test functions into buckets. It is static, so it only produces *candidates*, never verdicts

| bucket | meaning |
| --- | --- |
| `dead_skip` | unconditionally skipped, so it can never fail |
| `swallowed_failure` | the assertion sits in a `try` whose handler swallows the failure |
| `trivial_assert` | `assert True`, `assert <literal>`, or a value compared with itself |
| `mock_tautology` | both sides of the comparison are values the test configured on a mock |
| `no_assert` | no `assert`, `pytest.raises`/`fail`, or `assert_*` call anywhere in the body |

`no_assert` is the noisiest bucket: a test that asserts by not raising is legitimate. That is what Stage B is for

Commands:

```bash
python tests/vacuous_tests/inventory.py --report
python tests/vacuous_tests/inventory.py --check            # CI ratchet
python tests/vacuous_tests/inventory.py --update-baseline  # after a cleanup
python tests/vacuous_tests/inventory.py --areas                 # candidates per area
python tests/vacuous_tests/inventory.py --queue 15 --todays-area # the daily batch
python tests/vacuous_tests/inventory.py --queue 15 --area tests/test_litellm/proxy
```

`--todays-area` picks one area per day from the ranked list, rotating by date. That keeps every PR inside one owner's area and needs no state file, so two runs on the same day cannot disagree about where they are working

`inventory_baseline.json` names every known candidate per file and bucket, not just how many there are. `--check` fails on any candidate it does not already name, so fixing one test does not open a slot for a new vacuous one in the same file. The cost is that renaming or moving a known candidate also fails the check, which is a one-command fix: if the failure is a rename, or a deliberate assert-by-not-raising test whose docstring says so, regenerate the baseline

The scan matches pytest's own collection rules rather than a narrower guess, so `test*` names, not only `test_*`, and tests defined inside module-level `if` or `try` blocks are seen too, while nested helpers and anything under `if __name__ == "__main__"` are not, since pytest never binds those on import. One hole is left on purpose: a new vacuous test that reuses the exact name of a fixed one keys to the same baseline entry and passes, which takes deliberate effort and reads as such in review

## Stage B: does the test actually have teeth

`mutation_probe.py` decides. For one test it runs the test under coverage, subtracts the coverage floor of a no-op test in the same directory (so import-time lines are not counted), mutates only the lines the test itself executed, and re-runs the test against each mutant

```bash
python tests/vacuous_tests/mutation_probe.py "tests/x/test_y.py::test_z" --record
```

Mutants never touch the working tree: each one is written into a temp directory that symlinks the whole repo except the mutated file, and pytest runs with that directory as cwd, so a killed or crashed probe cannot leave mutated source behind

Verdicts: `vacuous` (survived every mutant), `not_vacuous` (a mutant killed it, recorded in `verified_not_vacuous.json` so it is never re-flagged), `dead` (skipped in this environment), `already_failing`, or `inconclusive`

Only a `vacuous` verdict authorizes editing a test

`not_vacuous` is a floor, not a compliment: it means the test notices when the code it covers changes behaviour or starts raising, which for an assert-by-not-raising test is all it ever claimed. Reviewing `verified_not_vacuous.json` is still worthwhile, since a cleared test can be weaker than its name suggests

## Fixing a vacuous test

Refactor first: add the assertion the test's own name and docstring imply, then re-run the probe and confirm the mutant that used to survive now dies. Delete only when the behaviour is provably covered somewhere else, and cite that test id. When neither is possible, leave it alone and record why: a human should look at it

## Not becoming flaky

`flake_gate.py` runs every touched test five times under different hash seeds, runs the owning file as a whole, and statically rejects sleeps, wall-clock reads, unseeded randomness, and unmocked network calls

```bash
python tests/vacuous_tests/flake_gate.py "tests/x/test_y.py::test_z"
```

## Guardrails

`guardrails.py` checks the daily automation's own PRs, so it fails by design on the PR that introduced this directory. It reads the committed diff (`base...HEAD`), not the working tree

It runs against the diff and rejects anything that games the metric: files outside `tests/`, edits to `conftest.py` or CI config, edits to this directory's own logic, test removals without a citation, and assertion counts dropping without tests being removed

```bash
python tests/vacuous_tests/guardrails.py --base origin/litellm_internal_staging
python tests/vacuous_tests/guardrails.py --base origin/litellm_internal_staging --allow-removals removals.json
```

Each removal needs its own entry, keyed by the removed test id:

```json
{"tests/x/test_y.py::test_z": "tests/x/test_y.py::test_z_rejects_bad_input covers this"}
```

## Daily automation

The scheduled run pulls the next batch from `--queue 15 --todays-area`, probes each candidate, fixes only the confirmed ones, clears the rest into `verified_not_vacuous.json`, runs the flake gate and the guardrails, then opens a single PR capped at 15 tests in one area. Anything it cannot fix honestly is reported rather than patched

It stops rather than lowering the bar: if fewer than three candidates survive probing it opens no PR that day, and if three or more of its own PRs are still open it skips the run entirely
