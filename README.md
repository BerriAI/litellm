# Prompt caching request table verification

PR: https://github.com/BerriAI/litellm/pull/42055

Feature baseline: f49fd22875b10d968fdefda1a5c3d6e0b7717f2c
Verified fix: 94b2fd827b4ea2678fa88ac0788e948f3b899348

Local proxy16540 and Postgres15540, dashboard16541. Real Anthropic calls used the gateway account; provider responses were not mocked. `write.json`, `read.json`, and `client.json` are the reproduction payloads

The current run returned write4917/read0 with negative net savings, then read4902/write15 with positive savings, then client caching read4902 with injection not recorded. `requests.json` records the current real results

## Cursor regression

On superseded643fde0c3e, the live endpoint returned HTTP200 for the first page and HTTP500 following next_cursor, with Prisma `incorrect binary data format in bind parameter4`. `binding-before-first.json`, `binding-before-second.json`, and `pagination-before.png` capture this reproduction

On94b2fd827b, the same cursor returns HTTP200. Timestamp parameters now bind as text before PostgreSQL timestamp conversion, preserving microsecond precision. The existing mapped tests now execute through actual Prisma rather than a psycopg query adapter; old production code fails the regression with the same bind error, and145 focused tests pass with the fix

`binding-verification.json` records complete current traversals: all66 rows in pages50+16; injected64 in50+14; hits5 in five one-row pages. Each traversal exactly matches its unpaginated result without duplicated or missing request IDs. The55 `pagination-qa-*` records are clearly labeled local synthetic fixtures only, used to exercise the dashboard's50-row limit; their savings are unavailable

Open http://localhost:16541/cost-optimization/, select Prompt Caching, select All caching, click Next then Previous. Page2 shows16 rows and disables Next; Previous restores50 rows on page1. `pagination-after.png` captures the successful page2
