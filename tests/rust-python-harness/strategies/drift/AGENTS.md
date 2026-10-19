# Drift

Compare committed Python and Rust changes using this folder's `drift.json`. Run locally without provider calls or CI integration

```sh
uv run python -m tests.rust-python-harness run drift --target HEAD --output /tmp/litellm-drift.json
```

Paths are repository-relative files or directory prefixes ending in `/`. Keep unmapped changes and uncommitted files visible in the report. Commit manifest renames separately from mapping edits so Git can follow the old watch paths

`checkpoint_sha` is a comparison snapshot. `checkpoint_state` records `unknown`, `drifted`, or `aligned` at that snapshot. Never update either automatically. `no_changes` does not imply alignment

A mapping may carry its own `checkpoint_sha` and `checkpoint_state`, recording the commit at which its Rust paths were reviewed against its Python paths. `mapping_checkpoints` in the report lists, for each such mapping, the commits and patches touching its paths since that checkpoint. Port those changes to the other side, then move the mapping's `checkpoint_sha` forward by hand in the same PR

```sh
uv run python -m tests.rust-python-harness run drift --output /tmp/litellm-drift.json
jq '.mapping_checkpoints[] | select(.id == "<mapping id>")' /tmp/litellm-drift.json
```

Use `--since` for an exploratory comparison and `--manifest` for another mapping file. Omit `--output` to print JSON. Fetch remote refs before comparing them. This strategy has its own options and does not run through `run all`
