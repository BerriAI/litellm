# Usage dashboard demo (synthetic data)

A local stack for walking through Usage > Overview and the Claude Code client panel with a year of synthetic history. It builds LiteLLM from this checkout, runs two proxy workers that share one dedicated Redis, and keeps everything in its own Postgres volume (`litellm_usage_demo_postgres`) under the fixed Compose project `litellm-usage-demo`. That volume is ephemeral and owned by this demo. Never point the stack or the SQL scripts at a database you care about. Both scripts check that Postgres runs with `cluster_name=litellm_usage_demo`, which only this Compose file sets, and refuse to run anywhere else.

No provider credentials are needed and no model is ever called. The seed writes daily aggregate rows directly into `LiteLLM_DailyUserSpend` (what the Usage page reads), `LiteLLM_DailyTagSpend` (the Claude Code user-agent tag the client panel reads) and `LiteLLM_DailyGatewayRequests` (request tallies under the route `usage-demo:/chat/completions`). Because it skips the spend log and the aggregation writer, the demo checks the read path and the UI only. It is not evidence that request ingestion or aggregation is correct. All seeded keys, users and model groups start with `usage-demo-` or `demo-`, so they are easy to recognise as synthetic.

Every service rotates its Docker logs at 10 MB with three files kept.

Pinned images are `postgres:16.6` and `redis:7.4.2`. The proxy image is built locally as `litellm-usage-demo:local` and is never pulled.

## Start, seed, open

Run from the repository root. The master key only lives in your shell and in the containers.

```sh
export USAGE_DEMO_MASTER_KEY="sk-$(openssl rand -hex 16)"
export COMPOSE_FILE=litellm/deploy/compose/usage-demo/docker-compose.yml
docker compose up -d --build --wait
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U llmproxy -d litellm -f /seed/seed_usage_demo.sql
```

The seed prints the number of demo rows, the first and last day, total spend and total tokens. Running it again replaces exactly the same rows, so the output stays the same on the same day.

Open http://localhost:4000/ui, sign in as `admin` with the value of `USAGE_DEMO_MASTER_KEY`, and go to Usage. If port 4000 is taken, set `USAGE_DEMO_PORT=4100` (or any free port) before `up` and use that port instead.

To confirm both workers are serving, `docker compose logs litellm | grep "Started server process"` should list two different PIDs.

## What the synthetic data contains

Twelve calendar months ending today in UTC, so the current month is partial. A partial month only means the range stops inside it. It says nothing about whether every day in it is fully aggregated. Eleven model groups across OpenAI, Anthropic, Bedrock, Gemini, Azure, Vertex AI and a subscription provider, which is more than the eight leaders the charts name, so the rest fall into Other. `demo-coding-assistant` starts in the fourth month and grows fast while `demo-legacy-completion` declines and stops after the eighth, which shows the model mix shifting. Prompt tokens include cache reads and writes as a subset. `demo-image-gen` has per-request spend with zero tokens and `demo-subscription-chat` has tokens with zero metered spend. The fifth month has no traffic at all, so it shows up as an empty month rather than disappearing.

## Manual QA checklist

Pick Last 12 months in the date picker (shortcut `12M`) unless a step says otherwise.

1. The Usage trend panel opens in Monthly buckets with the Tokens chart above the Spend chart. Both share the same month axis, the same legend, the same eight models plus Other, in the same stack order and colors.
2. Each month's total is printed above its bar. Hovering or keyboard-focusing a bar shows the month name, the exact total and each model's exact value, with spend to the cent or better and tokens as full integers.
3. The current month's tick reads like `Nov 1–12` and the note under the charts lists it as a partial period. The fifth month shows a zero total instead of being skipped.
4. Hover or focus the long `demo-eu-regional-assistant-...` legend entry. The full name appears. In the Contribution by model table it wraps instead of being cut off.
5. The table lists every model group, including the ones grouped as Other, each tagged In Other with the grey swatch. The Total row's tokens and spend match the sums of the bars in the charts.
6. `demo-image-gen` shows "Spend without tokens" in the Observed $ / 1M tokens column, and `demo-subscription-chat` shows "No metered spend". No row shows Infinity or NaN.
7. With Last 3 months selected, the Spend, Total Tokens and Avg cost / request tiles each show a change line such as `+18.4% vs Jun 1 – Aug 12`, which compares against the three months before with the same number of elapsed days in the last month. Month to date compares against the same elapsed days of last month, and Last 7 days against the 8 days before it. With 12M the previous twelve months hold no seeded data, so the tiles read New, the expected wording for a zero baseline.
8. Switch to Weekly and Daily. Bars and totals regroup, colors and model order stay the same, and totals above bars disappear once there are too many bars to label.
9. Switch Chart view to Requests. One stacked requests chart replaces the pair, with the same models and colors.
10. Switch the dashboard theme to dark. Charts, legend, notes and table stay readable.
11. Narrow the window to phone width. Controls wrap, charts keep full width and the table scrolls horizontally.
12. Export data from the Overview tab still downloads the same range and scope as the page shows.
13. Pick a user from the user filter (for example `usage-demo-user-b`). Stats, charts and table only show that user's five models from the research and coding keys, and the comparison follows the same filter.
14. Open the Claude Code client (client picker in the header, or its row under Top agents) with the all-users scope and 12M. The panel shows the same paired Tokens over Spend monthly charts and contribution table for `demo-coding-assistant` and `demo-subscription-chat`, whose totals match the panel's summary tiles. The subscription model contributes tokens with no metered spend.
15. In that panel, a person or team scope reads individual spend logs, which this demo does not seed, so it shows no Claude Code usage. That is expected here and not a regression.
16. Open the Models, Keys and Endpoints tabs. They behave as before and fire no previous-period request.

## Reset and clean up

Remove the usage-demo rows and keep the stack running:

```sh
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U llmproxy -d litellm -f /seed/reset_usage_demo.sql
```

Stop the stack and delete this project's containers, its volume and its locally built image, and nothing else:

```sh
docker compose down --volumes --rmi local
unset COMPOSE_FILE USAGE_DEMO_MASTER_KEY
```
