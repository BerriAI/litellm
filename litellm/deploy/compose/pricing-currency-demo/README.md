# Pricing currency demo

A local stack that shows custom deployment prices entered in EUR and HUF with a manual USD rate. It builds the gateway and dashboard from this checkout and runs PostgreSQL, a dedicated Redis, two gateway workers and a mock upstream. Every model call goes to the mock, which always reports 1000 prompt tokens and 500 completion tokens, so no provider key is needed and nothing is billed. All traffic and spend in this demo is synthetic, so treat it as evidence of LiteLLM's accounting, not of any model

Spend, budgets and exports stay in USD. The rate is typed by an admin and never fetched

## Fixtures

| Deployment | Where it is configured | Entered prices per 1M tokens | Rate | USD per 1M tokens | USD per mock call |
| --- | --- | --- | --- | --- | --- |
| `demo-usd-200` in `demo-currency-cheapest` | `proxy_config.yaml`, `litellm_params` | USD 2.00 in, 4.00 out | | 2.00 in, 4.00 out | 0.004 |
| `demo-huf-720` in `demo-currency-cheapest` | `proxy_config.yaml`, `litellm_params` | HUF 720 in, 1440 out | 0.003 | 2.16 in, 4.32 out | 0.00432 |
| `demo-eur-180` in `demo-currency-cheapest` | `/model/new` from `seed.py` | EUR 1.80 in, 3.60 out | 1.20 | 2.16 in, 4.32 out | 0.00432 |
| `demo-huf-model-info` | `proxy_config.yaml`, `model_info` | HUF 720 in, 1440 out | 0.003 | 2.16 in, 4.32 out | 0.00432 |
| `demo-eur-mixed` | `/model/new`, input in `litellm_params`, output in `model_info` | EUR 1.80 in, 3.60 out | 1.20 | 2.16 in, 4.32 out | 0.00432 |
| `demo-huf-db` | `/model/new`, `model_info`, editable in the UI | HUF 720 in, 1440 out | 0.003 | 2.16 in, 4.32 out | 0.00432 |

`demo-currency-cheapest` uses cost-based routing. Compared as raw numbers, EUR 1.80 looks cheaper than USD 2.00, but the router compares USD, so `demo-usd-200` serves the group

## Run it

You need Docker with Compose v2 and `openssl`. Run these from this directory

```bash
./generate-env.sh
docker compose up --build -d
docker compose --profile seed run --rm seed
```

`generate-env.sh` writes demo-only random keys to `.env`, which git ignores. The gateway listens on `127.0.0.1:4000` only (set `DEMO_PORT` to change it), and PostgreSQL, Redis and the mock are not published at all. Startup makes no model calls and opening the UI makes none either

The seed script creates the three API deployments, a key `demo-currency-budget-key` with a $0.03 USD budget, a key `demo-currency-pricing-key` without a budget, and six synthetic requests on the budget key that spend $0.02528. It is safe to rerun. Models and keys are created only when missing, and each request is recorded on the budget key right after it is answered, so a rerun sends only the requests not yet recorded. To start over, use the cleanup below

The image is built with the CI Rust profile by default, the same one the verification used. Set `DEMO_MATURIN_PEP517_ARGS=""` before `docker compose up --build` for the release profile

`demo-huf-db` points at a mock model that also reports a provider cost of $9.99. A deployment with custom prices bills its own prices, so its requests still cost $0.00432

## Walk through it

Load the demo keys into your shell first. The pricing checks use `DEMO_PRICING_KEY`, which has no budget, and the budget check at the end uses `DEMO_VIRTUAL_KEY`

```bash
set -a; . ./.env; set +a
call() {
  curl -s -D - -o /dev/null http://127.0.0.1:4000/v1/chat/completions \
    -H "Authorization: Bearer $1" -H "Content-Type: application/json" \
    -d "{\"model\": \"$2\", \"messages\": [{\"role\": \"user\", \"content\": \"synthetic\"}]}" \
    | grep -i -E 'HTTP/|x-litellm-model-id|x-litellm-response-cost'
}
```

1. Open http://127.0.0.1:4000/ui and log in as `admin` with `UI_PASSWORD` from `.env`
2. Run `call "$DEMO_PRICING_KEY" demo-currency-cheapest`. It prints `demo-usd-200` and `0.004`, because USD 2.00 is cheaper than EUR 1.80 at 1.20 once both are in USD
3. Open Models + Endpoints and select `demo-eur-180`. The pricing summary shows Input $2.16/1M tokens and Output $4.32/1M tokens, then "Prices entered in EUR at 1 EUR = $1.20", the entered EUR prices, and "Spend and budgets are tracked in USD."
4. Click Edit Settings. Price currency shows EUR and USD per 1 EUR shows 1.20. The USD preview lists each price with its unit. Clear the rate and click Save Changes, and the form asks for a rate instead of saving
5. Set USD per 1 EUR to `1.10`. A notice says the existing input and output prices are re-read at the new rate, and the preview shows USD 1.98 and USD 3.96 per 1M tokens. Click Save Changes, then reload the page. The currency and rate read back as entered, and the summary shows $1.98 and $3.96
6. Run `call "$DEMO_PRICING_KEY" demo-currency-cheapest` again. It now prints `demo-eur-180` and `0.00396`, since EUR at 1.10 is USD 1.98 per 1M input tokens
7. Open Logs. The seed's rows still show $0.004 for `demo-currency-cheapest` and $0.00432 for the other groups, and only the request from step 6 shows $0.00396. A row's metadata shows `cost_breakdown.pricing_currency`, `usd_per_currency_unit` and `pricing_revision`, and the revision differs before and after the edit
8. Select `demo-huf-db` and click Edit Settings. Switch Price currency to EUR without touching the prices. The notice lists both prices as re-read in EUR and the preview shows the new USD values. Click Cancel
9. Run `docker compose restart gateway`, wait for `docker compose ps` to show it healthy, and repeat step 6. The cost is still `0.00396`, so the entered values were kept and converted once
10. Open Virtual Keys and select `demo-currency-budget-key`. Its spend of $0.02528 and its $0.03 budget are in USD. Run `call "$DEMO_VIRTUAL_KEY" demo-currency-eur-mixed` a few times. The first two succeed at $0.00432 each, and once the spend passes $0.03 the next request is refused with a budget error. Spend is shared through Redis, so allow a few seconds between calls for both workers to see it

## Clean up

```bash
docker compose --profile seed down -v
docker image rm litellm-pricing-currency-demo:local
rm .env
```

This removes only this project's containers and network, its `litellm-pricing-currency-demo-postgres` volume, which is labeled disposable, and the locally built demo image. Docker's build cache is kept, and `docker builder prune` clears it if you want the space back

## Configuring currency on your own deployments

Set `pricing_currency` to `USD`, `EUR` or `HUF`, and for EUR or HUF also `usd_per_currency_unit`, the USD value of one unit of that currency. Put them on `litellm_params` or `model_info`, next to the custom prices they apply to. Every custom price on the deployment, in either place, is read in that currency, while thresholds, multipliers and cost map prices are left as they are. Leaving both settings out keeps today's USD behavior
