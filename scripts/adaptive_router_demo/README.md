# Adaptive Router — Live Demo

A 5-minute demo of LiteLLM's adaptive router learning, in real time, that
the smart model wins for code while the fast model is fine for facts.

```
┌─ traffic.py ──┐    ┌─ litellm proxy ──────────┐    ┌─ dashboard.html ─┐
│ synthetic     │──▶│  adaptive_router strategy │──▶│  bandit bars +    │
│ chat sessions │    │  /adaptive_router/state  │    │  cost meter +     │
└───────────────┘    └──────────┬───────────────┘    │  activity log     │
                                │                     └───────────────────┘
                      ┌─────────▼───────────┐
                      │    chat.html        │
                      │  interactive chat   │
                      │  with preset        │
                      │  scenarios          │
                      └─────────────────────┘
```

## Files

| File | What it does |
|---|---|
| `dashboard.html` | Live bandit dashboard — polls `/adaptive_router/state` every 500ms |
| `chat.html` | Interactive chat with preset scenarios — sends real requests through the router |
| `traffic.py` | Synthetic traffic generator — drives labeled sessions for automated demo |

## What you're watching

- **Bandit posteriors** — one Beta(α, β) bar per `(request_type, model)`
  cell. Bars fill up as α grows from positive feedback signals.
- **Pick share** — softmax estimate of how often the router would currently
  pick each model for that request type.
- **Cost meter** — total spend so far compared to "always use the most
  expensive model". The savings line is the headline number.
- **Activity log** — every signal that moves the bandit, in real time.

## 1. Start the proxy

The repo ships with a working example config:

```bash
export OPENAI_API_KEY=sk-...     # underlying models hit OpenAI
uv run litellm \
    --config litellm/proxy/example_config_yaml/adaptive_router_example.yaml \
    --port 4000
```

`DATABASE_URL` is optional — the proxy falls back to a bundled Neon dev DB.
Wait ~15s until you see `Application startup complete`.

## 2. Chat interactively with the router

Open `chat.html` in a browser (same `file://` or `python3 -m http.server` approach as the dashboard):

- Click **Connect** after filling in the proxy URL and API key.
- Pick a preset scenario:
  - **🐛 Debug my code** — paste broken code and get a fix
  - **💡 Brainstorm a feature** — ideate on a product capability
  - **📚 Explain a concept** — get a clear technical explanation
  - **✍️ Write something** — draft emails, docs, or any prose
- A starter message is pre-filled — edit it or send as-is.
- Each response shows which model the router picked and the inferred request type (from the `x-litellm-adaptive-router-model` and `x-litellm-request-type` response headers).
- A sidebar gate indicator tells you when the session has accumulated enough messages for the bandit to start updating (4+ turns).

> **Note on headers:** The model/type headers are only readable in the browser if the proxy sets `Access-Control-Expose-Headers`. LiteLLM defaults to exposing them. If the info panel shows `check dashboard`, the router still works — you can verify picks in `dashboard.html`.

## 4. Open the dashboard

The dashboard is a single static HTML file. Either:

- **Easy:** double-click `dashboard.html`. Most browsers will load it from
  `file://` and the LiteLLM proxy's CORS defaults (`*`) will accept it.
- **If your browser blocks `file://` fetches:**

  ```bash
  cd scripts/adaptive_router_demo
  python3 -m http.server 8080
  ```

  Then open <http://localhost:8080/dashboard.html>.

In the connect bar, fill in:

- **Proxy URL:** `http://localhost:4000`
- **Master Key:** the `master_key` from your config (`sk-1234` in the example).

Click **Connect**. The dashboard polls `GET /adaptive_router/state` every
500ms (admin-only endpoint, returns one snapshot per configured router).

## 5. Drive synthetic traffic

In a second terminal:

```bash
uv run python scripts/adaptive_router_demo/traffic.py \
    --proxy-url http://localhost:4000 \
    --api-key   sk-1234 \
    --router    smart-cheap-router \
    --rounds    100 \
    --rate      0.5
```

What it does:

- Picks a random `(request_type, prompt)` per round from a small labeled corpus.
- Sends a 5-message conversation (passes the `SIGNAL_GATE_MIN_MESSAGES=4` gate
  in one round-trip) so the post-call hook runs and updates the bandit.
- Reads the `x-litellm-adaptive-router-model` response header to see what
  the router picked.
- Rolls Bernoulli against a hard-coded oracle:
  ```
  code_generation : smart=0.92  fast=0.35
  factual_lookup  : smart=0.90  fast=0.85
  writing         : smart=0.85  fast=0.55
  ```
- On success → sends a follow-up engineered to match the satisfaction
  regex (and re-classify into the same type). Bandit cell gets +α.
- On failure → sends a neutral follow-up. No signal fires.

After 50–80 rounds you'll see `code_generation` decisively favor `smart`
while `factual_lookup` stays near a coin flip — the router learned the
asymmetry from the oracle.

## Tuning knobs

| Knob | Where | What changes |
|---|---|---|
| Quality vs. cost weight | `adaptive_router_config.weights` in proxy yaml | Bias toward quality or savings |
| Per-cell cold-start mass | `litellm/router_strategy/adaptive_router/config.py` `COLD_START_MASS` | How long until the prior is overwritten |
| Avg tokens per request | dashboard input box | How the cost meter estimates spend |
| Oracle | `traffic.py` `ORACLE` dict | Which model "should" win for which type |
| Sessions to drive | `--rounds` | Total learning budget |
| Throttle | `--rate` | Seconds between sessions |

## Multi-router

If your proxy has more than one `auto_router/adaptive_router` deployment,
the dashboard shows a router dropdown above the bars. Each router is
independent; the cost meter is per-router (and resets when you switch).

## Troubleshooting

- **"Disconnected" / HTTP 401 in the dashboard** — wrong master key.
- **HTTP 403** — your key isn't `proxy_admin`. The state endpoint is
  admin-only. Use the master key.
- **HTTP 404 from `/adaptive_router/state`** — proxy started, but no
  `auto_router/adaptive_router` deployment is in the model list.
- **Bars don't move** — check the proxy logs for `record_turn` activity.
  Common cause: requests are not including 4+ messages, so the signal
  gate skips them. `traffic.py` already builds 5-message conversations,
  so this only happens if you've changed the script.
- **Cost meter stays at $0** — your model deployments don't have
  `input_cost_per_token` set in `litellm_params`. Add it.
- **CORS error in the dashboard console** — set `LITELLM_CORS_ORIGINS=*`
  on the proxy (the default), or serve `dashboard.html` from
  `python3 -m http.server` instead of `file://`.
