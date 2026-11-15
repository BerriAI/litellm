# Prompt cache alerts demo

A local stack that shows the gateway noticing when Anthropic prompt caching stops working on a deployment, and the
Alerts page that reviews the result. It makes no model calls and needs no provider keys.

| Service | What it is |
|---|---|
| `proxy` | LiteLLM built from this checkout, image tagged with the commit, two workers |
| `db` | Postgres 16, holding the saved alerts in the demo-owned volume `litellm_prompt_cache_demo_postgres` |
| `redis` | Redis 7 with persistence off, shared by both workers for routing, coordination and the detector's streaks |
| `upstream` | A scripted Anthropic Messages API on the internal network |
| `notifier` | A Slack webhook stand-in that keeps the posts it receives, at <http://localhost:4101/received> |
| `driver` | Sends the demo requests through the proxy and checks the outcome |

`config.yaml` turns everything on the way a gateway would. It adds `prompt_cache_observations` to
`router_settings.optional_pre_call_checks`, sets `model_info.prompt_cache_observation: true` on three deployments,
points `router_settings` and `general_settings.coordination_redis` at the same Redis, and sets
`general_settings.prompt_cache_alerts` and `prompt_cache_alert_notifications`.

Nothing secret is checked in. The first `demo.sh up` writes a random master key, UI password and
`LITELLM_SALT_KEY` to `demo.env` in `$DEMO_STATE_DIR`, which defaults to
`~/.local/state/litellm-prompt-cache-demo`, readable only by you. Both workers read the same salt from it.

## Run it

From a clean checkout, since the image is pinned to the commit you have checked out:

```sh
litellm/scripts/prompt_cache_alerts_demo/demo.sh up
litellm/scripts/prompt_cache_alerts_demo/demo.sh run
```

`up` builds the proxy image and waits for every health check. `run` sends real chat completion requests through
the proxy, waiting after each one until the detector's Redis state shows it was applied:

- `claude-demo-unproven` gets three misses on a prefix that was never cached. Nothing is assessed and no alert opens
- `claude-demo-healthy` gets a proving cache write, a miss, a cache read and two more misses. The read resets the
  streak, so no alert opens
- `claude-demo-east` gets a proving cache write and three misses. One alert opens and the notifier receives one post.
  Three more misses find the same saved alert, so nothing new opens or posts

It prints the open prompt cache alerts and the post count, then exits non-zero if they are not exactly
`claude-demo-east` and one post.

Open <http://localhost:4100/ui/alerts?type=prompt-cache>, sign in as `admin` with the `UI_PASSWORD` from
`demo.env`, and open the alert to see its evidence. Dismiss, resolve and reopen work as for any other alert. Prompt cache setup is
described under **Manage alerts**.

## Reset and clean up

`demo.sh reset` removes the stack, its volume and `demo.env`, then starts it again with new credentials, so
`demo.sh run` can run once more.
`demo.sh logs` prints the last 200 lines from the proxy and both stand-ins. Container logs are capped at three
10 MB files each.

`demo.sh down` removes the containers, the network, the `litellm_prompt_cache_demo_postgres` volume and
`demo.env`. It touches nothing outside the `litellm-prompt-cache-demo` project and its state directory. The built image stays, so remove it once you are done:

```sh
docker image rm "litellm-prompt-cache-demo:$(git rev-parse HEAD)"
```

The ports are 4100 for the proxy and 4101 for the notifier, both bound to 127.0.0.1. Set `DEMO_PROXY_PORT` or
`DEMO_NOTIFIER_PORT` to change them.
