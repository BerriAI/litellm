---
name: litellm-setup
description: Set up a self-hosted LiteLLM proxy, the OpenAI-compatible LLM gateway, on this machine, from install through provider keys, config.yaml, a master key, optional Postgres and Admin UI, and a verified chat completion. Use when the user asks to set up LiteLLM, install the LiteLLM proxy, self-host an LLM gateway, or put one endpoint in front of OpenAI, Anthropic, Gemini, Azure, Bedrock, Ollama, or other providers. Not for using the LiteLLM Python SDK inside an app
---

# Set up a LiteLLM proxy

Take the user from nothing to a running LiteLLM proxy that has answered one chat completion. Run every command yourself unless a step says the user does it

Never print a secret. Provider keys, the master key, the salt key, the database URL, and virtual keys live only in `.env`, and `config.yaml` reads them through `os.environ/NAME`. Check that a value is there with `grep -c '^NAME=.' .env`, never with `cat .env` or `echo "$NAME"`. Load `.env` into a shell with this line, which exports each line as it is, so a `&` or `$` in a value stays literal, and first drops any database settings the shell already had and sets `LITELLM_MODE=PRODUCTION`, which stops `litellm` from also loading a `.env` it finds above its install folder, such as `~/.env`, so only this `.env` picks the database. The steps that need it start with it:

```bash
unset DATABASE_URL DIRECT_URL DATABASE_HOST; export LITELLM_MODE=PRODUCTION; while IFS= read -r line || [ -n "$line" ]; do case "$line" in [A-Za-z_]*=*) export "$line" ;; esac; done < .env
```

## 1. Ask once

Ask everything in one message, with defaults the user can accept by saying "go":

- Providers and models. Default is OpenAI with its newest flagship model, using the `OPENAI_API_KEY` in your environment when `[ -n "$OPENAI_API_KEY" ] && echo set` says it is there. Run that check before asking, and say in the question that the default uses that variable, naming it but never its value, since a key in your environment can belong to another account
- Whether they want virtual keys, teams, spend tracking, or the Admin UI. Those need Postgres, plain routing does not. Default is no
- uv (default) or Docker. With Postgres, Docker runs the repo's quickstart, which brings its own database, and uv needs a Postgres server they already have, with a new, empty database for LiteLLM, so someone with their own Postgres server picks uv
- Folder and port. Defaults are `./litellm-proxy` (`./litellm-gateway` for the quickstart) and 4000

Then follow the steps for their answers. Plain routing skips steps 3 and 6. The quickstart is steps 2, 3, 8, and 9. uv with their own Postgres skips step 3

In the same message, say that someone who picks uv and would rather answer prompts can run `litellm --setup` in a separate terminal window once LiteLLM is installed, not through Claude Code. Never run it yourself, since it needs a real terminal and without one prints "Setup cancelled." and writes nothing. Give them the command as `LITELLM_MODE=PRODUCTION <bin>/litellm --setup`, with `<bin>` replaced by the folder `uv tool dir --bin` prints, to run from the folder they want the config in. It writes `litellm_config.yaml` there with the keys inline and offers to start the proxy, which then listens on every network interface and stops with Ctrl+C in that window. When the user is done, run `chmod 600 litellm_config.yaml; grep -qx 'litellm_config.yaml' .gitignore 2>/dev/null || echo 'litellm_config.yaml' >> .gitignore` in that folder, since the wizard leaves the file readable by every account on the machine and nothing keeps it out of git, then skip to step 8 and run it against their proxy, reading the master key from that file into a shell variable without printing it

## 2. Check prerequisites

```bash
uv --version; docker info --format '{{.ServerVersion}}'; docker compose version; curl --version | head -1; openssl version; command -v python3
```

On native Windows, ask the user to run Claude Code inside WSL, since every command here is POSIX shell. Every path needs curl, openssl, and python3, so ask the user to install any that is missing. The uv path needs uv, which downloads a supported Python (LiteLLM supports 3.10 to 3.14) when the `--python` range in step 4 rules out the one on the machine. If uv is missing, offer to install it, and only after the user says yes run `curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh`, which leaves their shell profile alone (https://docs.astral.sh/uv/getting-started/installation/). It puts uv in `~/.local/bin`, so start every later command that runs `uv` or `litellm` with `export PATH="$HOME/.local/bin:$PATH";`. The Docker path needs a running Docker daemon, and the quickstart also needs Docker Compose v2. If `docker info` says `permission denied`, the user's account cannot use Docker without sudo. Offer the uv path, or ask them to add themselves to the `docker` group the way https://docs.docker.com/engine/install/linux-postinstall/ shows and start a new login session before you check again

Check the port is free with `curl -s -o /dev/null --max-time 2 http://127.0.0.1:4000; echo $?`, which prints `7` (could not connect) only when nothing listens there. Don't use `lsof` for this, since it misses ports that root or Docker hold. If it is taken, pick another port and use it everywhere below, because a proxy asked for a busy 4000 silently moves to a random port

## 3. Postgres with Docker: run the quickstart

The repo's quickstart starts the gateway, Postgres, and the Admin UI in Docker, listening on this machine only, and puts a `litellm-gateway` command in that folder that stops, starts, updates, or removes them. It asks nothing when it has no terminal. It writes the master key, salt key, and database password to its own `.env` and never prints them. Download it, then run it with the folder and port from step 1. It returns once the gateway is ready:

```bash
curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/quickstart.sh -o quickstart.sh
LITELLM_DIR="$PWD/litellm-gateway" LITELLM_PORT=4000 sh quickstart.sh --yes
cd litellm-gateway
```

Models live in the database here, so add each one through the API instead of a `config.yaml`, naming it the way step 5 says. Pipe the request body in, so the provider key never shows up in output or on a command line. Unless the user agreed to the key in your environment, as step 5 says, ask them to add the line `OPENAI_API_KEY=<their key>` to this `.env` themselves. Run the loader line first:

```bash
python3 -c 'import json, os, sys; json.dump({"model_name": "gpt-6.1-sol", "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": os.environ["OPENAI_API_KEY"]}}, sys.stdout)' \
  | curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:4000/model/new \
      -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" -d @-
```

Run that in one shell command that starts with the loader line, since the master key is in `.env`. `200` means the model was added. Then go to step 8

## 4. Install

uv:

```bash
uv tool install --python '>=3.10,<3.15' 'litellm[proxy]'
export PATH="$(uv tool dir --bin):$PATH"; litellm --version
```

With their own Postgres, install `'litellm[proxy,extra_proxy]'` instead, with the same `--python` range, which adds the Prisma client the proxy uses for the database. Running `uv tool install` again with the new extras replaces the old install. Start every later command that runs `litellm` with `export PATH="$(uv tool dir --bin):$PATH";`, so this install runs even when the shell has no `litellm` or an older one from a venv or `pip install litellm`. `uv tool update-shell` puts it on the PATH of new shells by editing the user's shell profile, so run it only after the user says yes

Docker:

```bash
docker pull docker.litellm.ai/berriai/litellm:main-stable
```

## 5. Write `.env` and `config.yaml`

Run every command from here on in the folder from step 1. Create `.env` readable only by the user, keep it out of git, and add a master key unless it already has one. The proxy refuses to start without one, and `sk-` is the format of every key it issues:

```bash
mkdir -p litellm-proxy && cd litellm-proxy
touch .env && chmod 600 .env; grep -qx '.env' .gitignore 2>/dev/null || echo '.env' >> .gitignore
grep -q '^LITELLM_MASTER_KEY=.' .env || printf 'LITELLM_MASTER_KEY=sk-%s\n' "$(openssl rand -hex 32)" >> .env
```

Copy a provider key that is already in your environment only after the user agreed to that variable, in their step 1 answer or when you ask now, and copy it without printing it, as in `printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" >> .env`. Otherwise ask the user to add the line `OPENAI_API_KEY=<their key>` to `.env` themselves and tell you when it is there. If they paste a key into the chat anyway, write it to `.env` and never repeat it. Keep one `NAME=value` per line with no quotes, so `docker --env-file` and the loader line read the same value

Confirm each model's current LiteLLM name in the cost map instead of trusting memory, since model names go stale fast. This prints the chat models of the provider named last (`openai`, `anthropic`, `gemini`, `azure`, `bedrock`, `vertex_ai`, and so on) the way `config.yaml` takes them, with the provider in front. It matches by prefix, so `bedrock` also lists the newer Bedrock models and `vertex_ai` lists every Vertex family. For Ollama, use a model from `ollama list` instead:

```bash
curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json | python3 -c '
import json, sys
for name, row in json.load(sys.stdin).items():
    provider = str(row.get("litellm_provider", "")) if isinstance(row, dict) else ""
    if provider.startswith(sys.argv[1]) and row.get("mode") == "chat":
        prefix = "bedrock" if provider == "bedrock_converse" else "vertex_ai" if provider.startswith("vertex_ai") else provider
        print(name if name.startswith(prefix + "/") else prefix + "/" + name)
' openai
```

In `config.yaml`, `model` is a name from that list, as in `openai/gpt-6.1-sol` or `bedrock/us.anthropic.<name>`, and `model_name` is what clients will send. If the folder already has a `config.yaml`, show the user its `model_name` lines and ask before replacing it. This example uses the newest GPT model in the cost map when this skill was written, so replace it with what the user picked:

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

## 6. Their own Postgres, on the uv path

Ask the user to create a new, empty database for LiteLLM on their server (`CREATE DATABASE litellm;`) and add `DATABASE_URL=postgresql://...` for it to `.env` themselves. The proxy creates its tables there on first start and stops on a database that already holds other tables. If that URL goes through a transaction pooler such as PgBouncer, they also add `DIRECT_URL` for the same database without the pooler, which the migrations use. Then add a salt key. It encrypts the provider credentials the proxy stores in the database and falls back to the master key when unset, so set it once and never change it:

```bash
grep -q '^LITELLM_SALT_KEY=.' .env || printf 'LITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)" >> .env
```

Generate the Prisma client once inside the tool's environment. A pip or uv install doesn't ship it generated (Docker images do):

```bash
tool="$(uv tool dir)/litellm"
schema=$("$tool/bin/python" -I -c 'import importlib.util, os; print(os.path.join(importlib.util.find_spec("litellm").submodule_search_locations[0], "proxy", "schema.prisma"))')
PATH="$tool/bin:$PATH" prisma generate --schema "$schema"
```

## 7. Start the proxy

Bind to `127.0.0.1` unless the user wants other machines to reach it

uv, in one shell command that starts with the loader line, since the installed `litellm` does not read `.env` by itself:

```bash
export PATH="$(uv tool dir --bin):$PATH"
nohup litellm --config config.yaml --host 127.0.0.1 --port 4000 > litellm.log 2>&1 &
echo $! > litellm.pid
```

Docker:

```bash
docker run -d --name litellm --env-file .env -v "$PWD/config.yaml:/app/config.yaml:ro" -p 127.0.0.1:4000:4000 \
  docker.litellm.ai/berriai/litellm:main-stable --config /app/config.yaml --port 4000
```

Wait on the health route, stopping early if the proxy exits. The first start with Postgres applies the database migrations and takes longer. Define `alive` for the path you took, `alive() { kill -0 "$(cat litellm.pid)" 2>/dev/null; }` for uv or `alive() { [ "$(docker inspect -f '{{.State.Running}}' litellm 2>/dev/null)" = true ]; }` for Docker, in the same shell command as the loop:

```bash
for i in $(seq 1 180); do curl -fsS http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1 && break; alive || break; sleep 1; done
curl -sS http://127.0.0.1:4000/health/readiness
```

Readiness reports `"db":"connected"` with Postgres and `"db":"Not connected"` without it. If the proxy never comes up, read the end of `litellm.log` (or `docker logs litellm`) and check the troubleshooting list

## 8. Verify with a chat completion

Use a `model_name` from `config.yaml` (or step 3), in one shell command that starts with the loader line:

```bash
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "Say hello in five words."}]}'
```

Show the user the reply in `choices[0].message.content`. An error here means the setup is not done, so fix it before handing off

## 9. Hand off

Tell the user, without printing any secret:

- The base URL is `http://127.0.0.1:4000`. Any OpenAI-compatible client works with it as the base URL and the master key or a virtual key as the API key
- The master key is `LITELLM_MASTER_KEY` in the folder's `.env`
- To stop it, `kill "$(cat litellm.pid)"` (uv), `docker stop litellm` (Docker), `<folder>/litellm-gateway stop` (quickstart), or Ctrl+C in its window (`litellm --setup`). To start it again, rerun the step 7 command, `docker start litellm`, or `<folder>/litellm-gateway start`. `update` and `uninstall` update or remove a quickstart install. Give the quickstart commands with the folder's full path, since a bare `litellm-gateway` is on PATH only when `~/.local/bin` is, and can belong to an earlier install
- With Postgres, the Admin UI is at `http://127.0.0.1:4000/ui` (user `admin`, password the master key), and `POST /key/generate` with the master key creates virtual keys with their own budgets and model lists. Without Postgres, both need step 3 or 6 first
- The Admin UI warns that the master-key login is a shared secret, which is fine while only this machine reaches the proxy. Before anyone else can reach it, invite an admin with their own password on the Internal Users page, then set `general_settings.disable_env_credential_login: true` in the proxy config
- To manage the running proxy from a coding agent (keys, teams, users, models, MCP servers), install `npx skills add BerriAI/litellm-skills -g`

## Troubleshooting

- `refused to start: no master key is set` means the loader line did not run in the shell command that started `litellm`, or `--env-file .env` is missing from `docker run`
- `a database URL is set but the prisma CLI is neither on PATH nor importable` means the install lacks `extra_proxy`, so rerun step 4 with it, or, on plain routing, that the command that started `litellm` did not start with the loader line
- `The Client hasn't been generated yet` or `Unable to find Prisma binaries` means the Prisma generate command in step 6 has not run
- `Incorrect API key provided` or another provider `AuthenticationError` means a provider key is wrong. Fix it in `.env` and restart, since the proxy reads the environment only at startup. On uv, stop it and rerun step 7. On Docker, `docker rm -f litellm` and rerun step 7, since `docker restart` keeps the environment the container was created with. On the quickstart the key is saved with the model, so delete the model and add it again: `GET /model/info` lists each model's `model_info.id`, and `POST /model/delete` with `{"id": "<id>"}` deletes it
- `Cannot automatically baseline this database` means `DATABASE_URL` points at a database that already holds tables LiteLLM did not create. Point it at a new, empty database
- `LLM Provider NOT provided` means a `model` in `config.yaml` (or step 3) lacks its provider in front. Use the name as the step 5 lookup prints it
- `Invalid model name passed in model=...` means the request's `model` is not a `model_name` in `config.yaml` or step 3. `GET /v1/models` lists the ones the key can use
- `No connected db.` on a proxy without Postgres means the bearer token is not the master key
- `Conflict. The container name "/litellm" is already in use` means an earlier run left a container. `docker start litellm` reuses it, or with the user's OK `docker rm -f litellm` and rerun step 7
- Inside Docker, `localhost` in `config.yaml` (an Ollama `api_base`, say) is the container itself. Use `host.docker.internal`, and on Linux also pass `--add-host=host.docker.internal:host-gateway`
