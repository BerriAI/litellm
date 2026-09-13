# live-pr-risk rig commands

Companion to the live-pr-risk skill, shipped in-repo at `.claude/skills/live-pr-risk/references/rig.md`. SKILL.md is the operating model; this is the runnable half.

Concrete commands for the base-vs-head A/B. Everything is namespaced so it can run alongside other sessions' rigs.

## 0. Namespace and pin the target

```bash
MAIN="${LITELLM_MAIN:-$HOME/Development/litellm}"   # primary checkout, not your session worktree
PR=<N>
NS="prrisk${PR}"                                 # one token per run

HEAD_SHA=$(gh pr view $PR --json headRefOid -q .headRefOid)   # full SHA, never reconstructed
BASE_REF=$(gh pr view $PR --json baseRefName -q .baseRefName)
git -C "$MAIN" fetch origin "$BASE_REF" && git -C "$MAIN" fetch origin pull/$PR/head
BASE_SHA=$(git -C "$MAIN" merge-base origin/$BASE_REF "$HEAD_SHA")

echo "A(base)=$BASE_SHA  B(head)=$HEAD_SHA"
```

Ports, derived from the namespace so two runs never collide. Assert free before binding, do not just hope:

```bash
OFF=$(( (PR % 400) * 2 ))
PG_A=$((15000+OFF)); PG_B=$((15001+OFF))
PX_A=$((16000+OFF)); PX_B=$((16001+OFF))
for p in $PG_A $PG_B $PX_A $PX_B; do
  lsof -iTCP:$p -sTCP:LISTEN >/dev/null && { echo "port $p busy, pick another offset"; exit 1; }
done
```

## 1. Two worktrees

```bash
WT_A=/tmp/${NS}-base
WT_B=/tmp/${NS}-head
git -C "$MAIN" worktree add --detach "$WT_A" "$BASE_SHA"
git -C "$MAIN" worktree add --detach "$WT_B" "$HEAD_SHA"
```

Worktrees do not carry gitignored files. Sync each (this venv also serves the live proxy and any local test run):

```bash
for WT in "$WT_A" "$WT_B"; do
  (cd "$WT" && uv sync --inexact --frozen --group proxy-dev --extra proxy --extra extra_proxy)
done
# dashboard only if the PR touches ui/
(cd "$WT_B/ui/litellm-dashboard" && npm ci)
```

Add any NEW extra the PR itself introduces to the head side, or the head venv will not contain the thing under test.

## 2. Changed surface and dependency graph

### 2a. What changed

Every grep in this section runs in the head worktree. `$MAIN` sits on whatever staging is today, so grepping it misses the symbols the PR adds and counts unrelated local edits as dependents.

```bash
cd "$WT_B"
git diff "$BASE_SHA".."$HEAD_SHA" --stat
# symbols whose definition line moved
git diff "$BASE_SHA".."$HEAD_SHA" -U0 | grep -E '^[-+]\s*(async def |def |class |[A-Z_][A-Z0-9_]* *=)'
# wire-contract surfaces
git diff "$BASE_SHA".."$HEAD_SHA" -- litellm/proxy/schema.prisma '*.yaml'
git diff "$BASE_SHA".."$HEAD_SHA" -U0 | grep -E '^[-+].*(BaseModel|TypedDict|Field\(|os\.environ)'
```

Record for each item WHICH property changed (signature / return / raises / default / side effect / timing / name). That drives which dependents matter.

### 2b. Direct references (the easy half)

```bash
SYM=<symbol>
grep -rn "$SYM" litellm/ enterprise/ tests/ ui/ \
  --include='*.py' --include='*.ts' --include='*.tsx' --include='*.yaml' --include='*.md'
```

### 2c. The references grep misses (run every one of these)

```bash
# string dispatch / registries: search the LITERAL, not the symbol
grep -rn "\"$SYM\"\|'$SYM'" litellm/ enterprise/ ui/
grep -rn "$SYM" litellm/constants.py

# subclass overrides and duck-typed implementers of a re-signed method
grep -rn "def $METHOD" litellm/ enterprise/ --include='*.py'

# kwargs pass-through chains that swallow signature changes
grep -rn "\*\*kwargs" $(git diff "$BASE_SHA".."$HEAD_SHA" --name-only -- '*.py')

# response-field consumers: dashboard, python client, generated schema
grep -rn "$FIELD" ui/litellm-dashboard/src --include='*.ts' --include='*.tsx'
grep -rn "$FIELD" litellm/proxy/client/
grep -rn "$FIELD" ui/litellm-dashboard/src/types/schema.d.ts

# config keys / env vars
grep -rn "$KEY" litellm/ enterprise/ --include='*.py' --include='*.yaml'

# user docs live in the SIBLING checkout, not this repo
grep -rn "$SYM\|$KEY" "${LITELLM_DOCS:-$HOME/Development/litellm-docs}/docs/" 2>/dev/null
```

Collect the results into one table: dependent path, its entrypoint, and which changed property it depends on.

### 2d. Which dependents are untested (this is the target list)

```bash
cd "$WT_B"
uv run --no-sync pytest tests/test_litellm/<mapped test paths> \
  --cov=litellm/<dependent module> --cov-report=term-missing -q
```

`term-missing` lists uncovered lines; any dependent call site in that list is untested through your change. Cross-check by name, since a test may cover the caller while stubbing the changed function:

```bash
grep -rn "<dependent_function>" tests/ | grep -v '\.pyc'
```

No hit, or a hit that mocks the changed symbol, means it belongs on the target list. Rank the list by reachability: HTTP surface, then SDK, then background jobs and CLI, then enterprise-only config.

### 2e. Secondary: package-level changes

Only if the diff moves a manifest. Do not read the `uv.lock` text diff (reflow noise hides transitive bumps); diff the resolved sets after both syncs:

```bash
(cd "$WT_A" && uv pip freeze | sort) > /tmp/${NS}-deps-base.txt
(cd "$WT_B" && uv pip freeze | sort) > /tmp/${NS}-deps-head.txt
diff /tmp/${NS}-deps-base.txt /tmp/${NS}-deps-head.txt
git -C "$MAIN" diff "$BASE_SHA".."$HEAD_SHA" -- \
  'Dockerfile*' docker/ ui/litellm-dashboard/package.json litellm/proxy/schema.prisma
```

Fold anything on a request path into the target list. This is a footnote to the code-dependency work.

## 3. Two Postgres instances

```bash
docker run --rm -d --name ${NS}-pg-a -p $PG_A:5432 \
  -e POSTGRES_DB=litellm -e POSTGRES_USER=llmproxy -e POSTGRES_PASSWORD=dbpassword9090 postgres:16
docker run --rm -d --name ${NS}-pg-b -p $PG_B:5432 \
  -e POSTGRES_DB=litellm -e POSTGRES_USER=llmproxy -e POSTGRES_PASSWORD=dbpassword9090 postgres:16

# a container that is up is not a server accepting connections; pushing early fails with connection refused
for c in ${NS}-pg-a ${NS}-pg-b; do
  until docker exec $c pg_isready -U llmproxy -d litellm >/dev/null 2>&1; do sleep 1; done
done

(cd "$WT_A" && DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:$PG_A/litellm" \
   uv run --no-sync prisma db push --schema litellm/proxy/schema.prisma)
(cd "$WT_B" && DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:$PG_B/litellm" \
   uv run --no-sync prisma db push --schema litellm/proxy/schema.prisma)
```

If the PR changes `schema.prisma`, the head-side push IS a test: capture whether it applies cleanly over an existing base-shaped DB, not just over an empty one. Point the head proxy at a copy of the base DB once to prove the migration path a real upgrade takes.

## 4. Credentials

Check before asking, ask before skipping, never fake:

```bash
set -a; source "$MAIN/.env"; set +a          # LITELLM_LICENSE, provider keys; note the AWS split identity below
env | grep -E 'OPENAI|ANTHROPIC|GEMINI|VERTEX|AWS|AZURE|LITELLM_LICENSE' | sed 's/=.*/=<set>/'
```

`set -a` is load-bearing: dotenv lines carry no `export`, so without it the keys live in your shell alone, both proxies start without them, and identical auth failures on base and head read as a clean A/B.

Probe each key the plan needs with a minimal direct call before building on it:

```bash
curl -sS https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  | jq -r '.error.message // "ok"'
```

`insufficient_quota` / "credit balance is too low" means billing-dead, not a proxy bug. Report it to the user immediately and pick a funded provider for the control leg.

Traps:

* `AWS_BEARER_TOKEN_BEDROCK` overrides SigV4 for bedrock calls and can point at a different account while `aws sts get-caller-identity` keeps reporting the SigV4 identity. `unset AWS_BEARER_TOKEN_BEDROCK` in the proxy env AND every verification shell for any SigV4 rig.
* Premium-gated surfaces: source `LITELLM_LICENSE` from `$MAIN/.env`. Patching `_license_check.is_premium` is the fallback only, and must be disclosed in the report.

Missing or dead credential -> ask the user by exact variable name and what it unlocks. If unavailable, the scenario goes in the report's "Not verified" section by name.

## 5. Launch both proxies

Same config file, same flags, different ports and DBs. `PYTHONPATH` first, and assert it took:

```bash
CFG=/tmp/${NS}-config.yaml
cat > "$CFG" <<'YAML'
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-4.1-mini
      api_key: os.environ/OPENAI_API_KEY
general_settings:
  master_key: sk-1234
YAML

launch () {  # launch <worktree> <pgport> <proxyport> <tag>
  cd "$1" || return 1
  export PYTHONPATH="$1"
  export DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:$2/litellm"
  export LITELLM_MASTER_KEY="sk-1234"
  uv run --no-sync python -c \
    "import litellm,os,sys; print('$4 loaded:', litellm.__file__); sys.exit(0 if litellm.__file__.startswith(os.getcwd()) else 1)" || return 1
  setsid nohup uv run --no-sync litellm --config "$CFG" --port "$3" --detailed_debug \
    > /tmp/${NS}-$4.log 2>&1 < /dev/null &
}
launch "$WT_A" $PG_A $PX_A base
launch "$WT_B" $PG_B $PX_B head
```

The `litellm.__file__` assertion is the whole ballgame: without it a worktree-launched proxy silently imports the main checkout and the A/B compares one tree against itself.

## 6. Run each scenario on both sides

Wrap every scenario so it records which tree answered:

```bash
scenario () {  # scenario <port> <tag> <name> <curl-args...>
  local port=$1 tag=$2 name=$3; shift 3
  echo "=== $name [$tag:$port] $(date -u +%H:%M:%S) ==="
  curl -sS -o /tmp/${NS}-$name-$tag.body -w 'status=%{http_code} total=%{time_total}s\n' \
    "http://127.0.0.1:$port$1" "${@:2}"
}
```

Then diff the pair, body and status and spend row:

```bash
diff <(jq -S . /tmp/${NS}-<name>-base.body) <(jq -S . /tmp/${NS}-<name>-head.body)

psql "postgresql://llmproxy:dbpassword9090@localhost:$PG_A/litellm" -c \
  'select model, spend, total_tokens, status from "LiteLLM_SpendLogs" order by "startTime" desc limit 5'
# same against $PG_B
```

Baseline scenarios worth running on every PR regardless of what changed:

* non-streaming `/chat/completions`
* streaming `/chat/completions` (compare chunk count and boundaries, not just the final text)
* `/v1/messages` (Anthropic shape)
* key create -> use -> `/key/info` -> delete -> confirm 401
* a router fallback (primary model deliberately broken)
* spend row written for each of the above
* `/health` and `/health/readiness`
* dashboard load at `/ui` when `ui/` or a JS dependency changed

## 7. Merge-ref check

The shipped artifact is the PR merged into current staging, not the head alone:

```bash
if git -C "$WT_B" merge --no-edit origin/$BASE_REF; then
  pkill -f "litellm --config /tmp/${NS}-confi[g].*--port $PX_B"   # it still serves the pre-merge import
  (cd "$WT_B" && uv sync --inexact --frozen --group proxy-dev --extra proxy --extra extra_proxy)
  launch "$WT_B" $PG_B $PX_B head
else
  git -C "$WT_B" merge --abort    # the conflict IS the finding; report it, leave the head proxy serving the PR head
fi
```

A conflict is news worth reporting, and it ends the step there: tearing the head proxy down and syncing a conflicted tree would make every later scenario answer for neither side. Restarting the head proxy on a clean merge is the point of the step: a long-lived process keeps serving the modules it imported at boot, so scenarios re-run against it answer for the pre-merge tree and staging drift never shows up.

On the clean-merge path, re-run the highest-risk scenarios against the restarted head proxy. A difference that appears only here is staging drift interacting with the PR, which is still a real ship risk and belongs in the report. On the conflict path there is no merged tree to run anything on, so the conflict is the whole result of this step: report it and report the merge-ref check as not run, never scenarios from the pre-merge head dressed up as a merge-ref pass.

## 8. Teardown (yours only)

```bash
docker rm -f ${NS}-pg-a ${NS}-pg-b
pkill -f "litellm --config /tmp/${NS}-confi[g]"      # self-excluding bracket, scoped to this run
git -C "$MAIN" worktree remove --force "$WT_A"
git -C "$MAIN" worktree remove --force "$WT_B"
```

Never a bare `pkill -f proxy_cli.py` or `pkill -f litellm` since that kills other sessions' proxies. Match the namespace or the port.
