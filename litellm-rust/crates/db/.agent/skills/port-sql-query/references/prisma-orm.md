# Prisma ORM calls

## 1. Capture the SQL Prisma sends

Run the exact call against a scratch database. It really executes, so never point it at a database you care about:

```bash
SCRATCH_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5544/litellm \
  uv run --no-sync python scripts/prisma_sql.py \
  'db.litellm_verificationtoken.update(where={"token": "t"}, data={"budget_limits": "{}"})'
```

It prints every statement with its parameters, minus `"public".` qualifiers and trace comments. Seed the rows the call needs first, or the capture shows the empty-result path only

## 2. Turn it into the shared file

- Drop Prisma-only padding: the `OFFSET $n` it adds with `0`, the `AND 1=1`, the `COUNT(*) FROM (SELECT ...) AS "sub"` wrapper
- Replace `"Table"."col"` with `col` when one table is involved
- List only the columns the caller uses, unless it hands the whole row on
- Keep what Prisma did implicitly
  - `@updatedAt` columns: Prisma sends a UTC client timestamp, so write `updated_at = now() AT TIME ZONE 'UTC'` for `timestamp` columns
  - `@default(uuid())` ids come from the Prisma client, not the database: pass the id or use `gen_random_uuid()`
  - `update` / `upsert` return the row: use `RETURNING <columns>` and `INSERT ... ON CONFLICT (...) DO UPDATE ... RETURNING <columns>` when the caller reads it
  - `Json` fields: Prisma parses a string payload into JSON. Cast the parameter, `$n::jsonb`, and pass the string
  - `include=` runs one extra `SELECT` per relation. Choose a JOIN or separate files on purpose and say why in the change

## 3. Change the Python side

- Put the call in a repository method if it is not already in one, and run the shared SQL there with `query_raw` / `execute_raw`
- Validate rows with the Pydantic model the method already returns. Raw results carry timestamps as ISO strings, and the model parses them
- Inside a Prisma transaction, use `tx.query_raw` / `tx.execute_raw`
- Callers that used attributes of the Prisma model object (`row.expires`) now get the Pydantic model or a dict. Check every caller
- Mocked tests that assert on `.table.update(...)` must follow the new call. Add a database-backed test that runs the old ORM call and the new SQL on the same seeded rows and compares the result and the stored row

## Behavior differences to check

- prisma-client-py `update` returns `None` for a missing row, it does not raise. `execute_raw` returns `0`
- Prisma wraps some writes in `BEGIN` / `COMMIT`. A single statement is atomic on its own, but several statements that must succeed together need a transaction
