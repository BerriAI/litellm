---
name: litellm-setup
description: Set up a self-hosted LiteLLM proxy, the OpenAI-compatible LLM gateway, on this machine, from install through provider keys, config.yaml, a master key, optional Postgres, and a verified chat completion. Use when the user asks to set up LiteLLM, install the LiteLLM proxy, self-host an LLM gateway, or put one endpoint in front of OpenAI, Anthropic, Gemini, Azure, Bedrock, Ollama, or other providers. Not for using the LiteLLM Python SDK inside an app
---

# Set up a LiteLLM proxy

Take the user from nothing to a running LiteLLM proxy that has answered one chat completion. Run every command yourself unless a step says the user does it

Never print a secret. Provider keys, the master key, the salt key, the database password, and virtual keys live only in `.env`, and `config.yaml` reads them through `os.environ/NAME`. Check that a value is there with `grep -c '^NAME=.' .env`, never with `cat .env` or `echo "$NAME"`

## 1. Ask once

Ask everything in one message, with defaults the user can accept by saying "go":

- Providers and models. Default is OpenAI with its newest flagship model
- uv (default) or Docker
- Whether they want virtual keys, teams, spend tracking, or the Admin UI. Those need Postgres, plain routing does not. Default is no
- Folder and port. Defaults are `./litellm-gateway` and 4000

In the same message, say that someone who would rather answer prompts can run `litellm --setup` in their own terminal after step 3. Never run it yourself, since it needs a real terminal and without one prints "Setup cancelled." and writes nothing. It writes `litellm_config.yaml` in the current folder with the keys inline (keep it out of git) and offers to start the proxy. When the user is done, skip to step 7 and run it against their proxy, reading the master key from that file into a shell variable without printing it

## 2. Check prerequisites

```bash
uv --version; docker info --format '{{.ServerVersion}}'; curl --version | head -1; openssl version
```

The uv path needs uv, which picks or downloads a supported Python on its own (LiteLLM supports 3.10 to 3.14). If uv is missing, offer to install it the way https://docs.astral.sh/uv/getting-started/installation/ shows for their OS, and run the installer only after the user says yes. The Docker path needs a running Docker daemon. Postgres in step 5 also uses Docker unless the user already has a database

## 3. Install

uv:

```bash
uv tool install 'litellm[proxy]'
litellm --version
```

With Postgres, install `'litellm[proxy,extra_proxy]'` instead, which adds the Prisma client the proxy uses for the database. Running `uv tool install` again with the new extras replaces the old install. If `litellm` is not found, run `uv tool update-shell` for new shells and call `"$(uv tool dir --bin)/litellm"` in this one

Docker:

```bash
docker pull docker.litellm.ai/berriai/litellm:main-stable
```

## 4. Write `.env` and `config.yaml`

Run every command from here on in the folder from step 1. Create `.env` readable only by the user, keep it out of git, and add a master key unless it already has one. The proxy refuses to start without one, and `sk-` is the format of every key it issues:

```bash
mkdir -p litellm-gateway && cd litellm-gateway
(umask 077; touch .env); grep -qx '.env' .gitignore 2>/dev/null || echo '.env' >> .gitignore
grep -q '^LITELLM_MASTER_KEY=.' .env || printf 'LITELLM_MASTER_KEY=sk-%s\n' "$(openssl rand -hex 32)" >> .env
```

For each provider key, if it is already in your environment (`[ -n "$OPENAI_API_KEY" ] && echo set`), copy it in without printing it, as in `printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" >> .env`. Otherwise ask the user to add the line `OPENAI_API_KEY=<their key>` to `.env` themselves and tell you when it is there. If they paste a key into the chat anyway, write it to `.env` and never repeat it. Keep one `NAME=value` per line with no quotes, so the shell and `docker --env-file` read the same value

Confirm each model's current LiteLLM name in the cost map instead of trusting memory, since model names go stale fast. This lists the chat models for one provider (`openai`, `anthropic`, `gemini`, `azure`, `bedrock`, `ollama`, and so on):

```bash
curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json \
  | python3 -c 'import json,sys; [print(k) for k,v in json.load(sys.stdin).items() if isinstance(v,dict) and v.get("litellm_provider")=="openai" and v.get("mode")=="chat"]'
```

In `config.yaml`, `model` is `<provider>/<name from the cost map>`, except that names which already start with the provider (like `gemini/...`) are used as they are. `model_name` is what clients will send. This example uses the newest GPT model in the cost map when this skill was written, so replace it with what the user picked:

```yaml
model_list:
  - model_name: gpt-6.1-sol
    litellm_params:
      model: openai/gpt-6.1-sol
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Add one entry per model. Providers that need more than a key, like Azure or Bedrock, list their `litellm_params` on their page under https://docs.litellm.ai/docs/providers. Point every secret at `.env` through `os.environ/NAME` the same way

## 5. Postgres, only when the user wants it

Skip this step for plain routing. If the user already has a Postgres, ask them to add `DATABASE_URL=postgresql://...` to `.env` themselves and go to the uv-only part at the end of this step

Otherwise add a database password, its URL, and a salt key to `.env`. The salt key encrypts the provider credentials the proxy stores in the database and falls back to the master key when unset, so set it now and never change it. Use host `localhost` for the uv path and `litellm-db` for the Docker path:

```bash
pw=$(openssl rand -hex 16); host=localhost
printf 'POSTGRES_PASSWORD=%s\nDATABASE_URL=postgresql://litellm:%s@%s:5432/litellm\nLITELLM_SALT_KEY=sk-%s\n' \
  "$pw" "$pw" "$host" "$(openssl rand -hex 32)" >> .env
```

Start Postgres in Docker. On the uv path the proxy reaches it through the published port, so check `lsof -nP -iTCP:5432 -sTCP:LISTEN` prints nothing first, and on the Docker path drop the `-p` flag:

```bash
docker network create litellm
set -a; . ./.env; set +a
docker run -d --name litellm-db --network litellm -p 127.0.0.1:5432:5432 \
  -e POSTGRES_USER=litellm -e POSTGRES_PASSWORD -e POSTGRES_DB=litellm \
  -v litellm-db:/var/lib/postgresql/data postgres:16
for i in $(seq 1 60); do docker exec litellm-db pg_isready -h 127.0.0.1 -U litellm >/dev/null 2>&1 && break; sleep 1; done
```

Keep `-h 127.0.0.1` in the readiness check. On first boot Postgres answers on its socket during setup and then restarts, so a check without it passes too early

On the uv path only, generate the Prisma client once inside the tool's environment. A pip or uv install doesn't ship it generated (Docker images do):

```bash
tool="$(uv tool dir)/litellm"
schema=$("$tool/bin/python" -I -c 'import importlib.util, os; print(os.path.join(importlib.util.find_spec("litellm").submodule_search_locations[0], "proxy", "schema.prisma"))')
PATH="$tool/bin:$PATH" prisma generate --schema "$schema"
```

## 6. Start the proxy

Check the port is free with `lsof -nP -iTCP:4000 -sTCP:LISTEN`, which prints nothing when it is. If it is taken, pick another port and use it everywhere below, because a proxy asked for a busy 4000 silently moves to a random port. Bind to `127.0.0.1` unless the user wants other machines to reach it

uv, in one shell command, since the installed `litellm` does not read `.env` by itself:

```bash
set -a; . ./.env; set +a
nohup litellm --config config.yaml --host 127.0.0.1 --port 4000 > litellm.log 2>&1 &
echo $! > litellm.pid
```

Docker, adding `--network litellm` when Postgres runs in Docker:

```bash
docker run -d --name litellm --env-file .env -v "$PWD/config.yaml:/app/config.yaml:ro" -p 127.0.0.1:4000:4000 \
  docker.litellm.ai/berriai/litellm:main-stable --config /app/config.yaml --port 4000
```

Wait on the health route. The first start with Postgres applies the database migrations and takes longer:

```bash
for i in $(seq 1 180); do curl -fsS http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1 && break; sleep 1; done
curl -sS http://127.0.0.1:4000/health/readiness
```

Readiness reports `"db":"connected"` with Postgres and `"db":"Not connected"` without it. If the proxy never comes up, read the end of `litellm.log` (or `docker logs litellm`) and check the troubleshooting list

## 7. Verify with a chat completion

Use a `model_name` from `config.yaml`:

```bash
set -a; . ./.env; set +a
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "Say hello in five words."}]}'
```

Show the user the reply in `choices[0].message.content`. An error here means the setup is not done, so fix it before handing off

## 8. Hand off

Tell the user, without printing any secret:

- The base URL is `http://127.0.0.1:4000`. Any OpenAI-compatible client works with it as the base URL and the master key or a virtual key as the API key
- The master key is `LITELLM_MASTER_KEY` in the folder's `.env`
- To stop it, `kill "$(cat litellm.pid)"` (uv) or `docker stop litellm` (Docker). To start it again, rerun the step 6 command or `docker start litellm`
- With Postgres, the Admin UI is at `http://127.0.0.1:4000/ui` (user `admin`, password the master key), and `POST /key/generate` with the master key creates virtual keys with their own budgets and model lists. Without Postgres, both need step 5 first
- To manage the running proxy from a coding agent (keys, teams, users, models, MCP servers), install `npx skills add BerriAI/litellm-skills -g`

## Troubleshooting

- `refused to start: no master key is set` means `.env` was not exported into the shell that started `litellm`, or `--env-file .env` is missing from `docker run`
- `a database URL is set but the prisma CLI is neither on PATH nor importable` means the install lacks `extra_proxy`, so rerun step 3 with it
- `The Client hasn't been generated yet` or `Unable to find Prisma binaries` means the Prisma generate command in step 5 has not run
- `Incorrect API key provided` or another provider `AuthenticationError` means a provider key in `.env` is wrong. Fix it and restart, since the proxy reads the environment only at startup
- `Invalid model name passed in model=...` means the request's `model` is not a `model_name` in `config.yaml`. `GET /v1/models` lists the ones the key can use
- `No connected db.` on a proxy without Postgres means the bearer token is not the master key
- `password authentication failed` from Postgres means an old `litellm-db` volume kept the password it was created with. Put that password back in `.env`, or with the user's OK run `docker rm -f litellm-db && docker volume rm litellm-db` and redo step 5
- Inside Docker, `localhost` in `config.yaml` (an Ollama `api_base`, say) is the container itself. Use `host.docker.internal`, and on Linux also pass `--add-host=host.docker.internal:host-gateway`
