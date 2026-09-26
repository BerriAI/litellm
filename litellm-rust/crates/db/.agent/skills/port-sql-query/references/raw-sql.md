# Raw SQL calls

## Static SQL

Copy the text as it runs today into the `.sql` file. Only change whitespace, add aliases a Rust identifier needs, and expand `*`. If the SQL came from a module constant, delete the constant once nothing uses it

Implicitly concatenated literals and constants built at import time are static too. Print them to get the exact text instead of retyping it:

```bash
uv run --no-sync python -c "import litellm.proxy.db.baseline_accounting as m; print(m._READ_PAGE)"
```

## Dynamic SQL

Rewrite it into one of the static forms in the SQL rules of `litellm-rust/crates/db/AGENTS.md`. On the Python side:

- A clause added only when a value is set: always pass the value, `None` when unset
- A list spliced into `IN (...)`: pass the list itself
- Rows spliced into `VALUES`: pass one list per column for `UNNEST`, or `json.dumps(rows)` for `jsonb_to_recordset`
- A template filled from a small mapping (one entry per table, a flag): one constant per entry, picked where the template used to be built

## Behavior checks

- Prisma raw calls return `jsonb` as `dict` / `list` and timestamps as ISO strings. Keep whatever validation the caller already does
- `execute_raw` returns the affected row count. Keep callers that branch on it working
- A rewrite that changes the SQL text beyond whitespace needs a database-backed test showing the old and new forms give the same rows or the same writes
