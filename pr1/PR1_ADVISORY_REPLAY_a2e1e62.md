# PR1 advisory UI replay: baseline eb8a2d224e75593899f6af418c0aa5d3b3ce3060 vs head a2e1e62893de4ddcf3a26155d8cda15abb955b88

All alert rows are seeded fixtures (PR1 has no detector) and no model call is made. Setup, config, seed SQL and API results are in PR1_REPLAY_f594c55.md and are unchanged: the head diff from f594c55059425e024a12200e8c3d0cbf00814b8e to a2e1e62893d touches only UI copy (`en.json`), two UI test files, the `consecutive_misses` Field description in `litellm/models/alert.py` and the generated comment in `schema.d.ts`. Same Postgres database (rows 1 to 5 and 7, row 2 dismissed), dedicated Redis on 6393, two proxy workers, `LITELLM_LICENSE` unset, production dashboard build of each SHA served through `LITELLM_UI_PATH`.

```bash
redis-server --port 6393 --save "" --daemonize yes
cd litellm/ui/litellm-dashboard && npm run build && export LITELLM_UI_PATH=$PWD/out && cd -
python litellm/proxy/proxy_cli.py --config config.yaml --port 4000 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver
```
Two worker PIDs served and the `"workers" flag is ignored` warning did not appear. Env var names: `DATABASE_URL`, `MASTER_KEY`, `LITELLM_SALT_KEY`, `LITELLM_UI_PATH`.

Captures: log in at `/ui/login` as `admin` with `MASTER_KEY`, open the URL, 1440x1000 viewport. The red outline is drawn in the browser around the changed text before capture.

| URL | Baseline eb8a2d224e7 | Head a2e1e62893d |
|---|---|---|
| `/ui/alerts` | "Could not load performance alerts." (`GET /alerts` 500) | first row "Prompt cache usage advisory for fixture-deployment-1", "3 observed requests to claude-sonnet-5-5 reported no cache usage for a prefix the provider had cached before. Delayed, missing or unlogged observations mean these requests are not necessarily consecutive and do not prove a caching outage." |
| `/ui/alerts?alert=7` | alert fails to load (`GET /alerts/7` 500) | same title and description, "Observed requests with no cache usage 3", "Earlier cache write that proved the prefix 2048 tokens", "Cache lifetime 5 minutes" |

Head `GET /alerts/7` returned 200 with `type=prompt_cache`, evidence as seeded, and the head proxy log has 0 tracebacks.

Dashboard checks at a2e1e62893d, exit 0 each: `npm run typecheck`, `npx prettier --check` and `npx eslint` on changed files, `node scripts/check-lint-budgets.mjs`, `npm run knip:ci`, `npm run test:types`, `npx vitest run src/components/Alerts`, `npm run build`.
