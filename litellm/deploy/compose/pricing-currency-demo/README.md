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

`demo-currency-cheapest` uses cost-based routing. Compared as raw numbers, EUR 1.80 looks cheaper than USD 2.00, but the router compares USD, so `demo-usd-200` serves the group

## Run it

You need Docker with Compose v2 and `openssl`. Run these from this directory

```bash
./generate-env.sh
docker compose up --build -d
docker compose --profile seed run --rm seed
```

`generate-env.sh` writes demo-only random keys to `.env`, which git ignores. The gateway listens on `127.0.0.1:4000` only (set `DEMO_PORT` to change it), and PostgreSQL, Redis and the mock are not published at all. Startup makes no model calls and opening the UI makes none either

The seed script creates the two API deployments, a virtual key `demo-currency-key` with a $0.03 USD budget, and six synthetic requests. It is safe to rerun, since it skips models and the key when they exist and sends the traffic only once

## Walk through it

Load the demo keys into your shell first

```bash
set -a; . ./.env; set +a
```

1. Open http://127.0.0.1:4000/ui and log in as `admin` with `UI_PASSWORD` from `.env`
2. Open Models + Endpoints and select `demo-eur-180`. The pricing summary shows Input $2.16/1M tokens and Output $4.32/1M tokens, then "Prices entered in EUR at 1 EUR = $1.20", the entered EUR prices, and "Spend and budgets are tracked in USD."
3. Click Edit Settings. The Price currency selector shows EUR and USD per 1 EUR shows 1.20. The USD preview lists each price with its unit
4. Change USD per 1 EUR to `1.10`. A notice says the existing input and output prices are re-read at the new rate, and the preview now shows USD 1.98 and USD 3.96 per 1M tokens. Click Save Changes
5. Reload the page. The currency and rate read back as entered, and the summary shows $1.98 and $3.96
6. Send a request to the routed group

   ```bash
   curl -s -D - -o /dev/null http://127.0.0.1:4000/v1/chat/completions \
     -H "Authorization: Bearer $DEMO_VIRTUAL_KEY" -H "Content-Type: application/json" \
     -d '{"model": "demo-currency-cheapest", "messages": [{"role": "user", "content": "synthetic"}]}' \
     | grep -i -E 'x-litellm-model-id|x-litellm-response-cost'
   ```

   Before step 4 this prints `demo-usd-200` and `0.004`. After step 4, EUR at 1.10 is USD 1.98 per 1M input tokens, cheaper than USD 2.00, so it prints `demo-eur-180` and `0.00396`
7. Open Logs. The rows the seed wrote still show $0.004 for `demo-currency-cheapest` and $0.00432 for the two other groups, and only requests after the edit show $0.00396. A row's metadata shows `cost_breakdown.pricing_currency`, `usd_per_currency_unit` and `pricing_revision`, and the revision differs before and after the edit
8. Open Virtual Keys and select `demo-currency-key`. Its spend and $0.03 budget are in USD. Send the curl from step 6 a few times. Once the key's spend reaches $0.03 the next request is refused with a budget error

To prove a restart keeps the entered values and converts them only once, run `docker compose restart gateway`, then repeat steps 5 and 6. The values and costs stay the same

## Clean up

```bash
docker compose --profile seed down -v --rmi local
rm .env
```

This removes only this project's containers, its PostgreSQL volume and the locally built demo image

## Configuring currency on your own deployments

Set `pricing_currency` to `USD`, `EUR` or `HUF`, and for EUR or HUF also `usd_per_currency_unit`, the USD value of one unit of that currency. Put them on `litellm_params` or `model_info`, next to the custom prices they apply to. Every custom price on the deployment, in either place, is read in that currency, while thresholds, multipliers and cost map prices are left as they are. Leaving both settings out keeps today's USD behavior
