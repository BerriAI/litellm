# How to refresh the numbers in this series

Quadrant: how-to. Written against `main` at `13118948` (2026-10-15)

Every page carries a "Verified on" commit. The structure changes slowly, but counts and line numbers drift every week. These commands regenerate the numbers the pages depend on. Run them from a fresh checkout of `main` after `git pull`

## Layer edges (pages 02 and the gap register)

```bash
S=$(mktemp -d)
python3 /mnt/project-files/invariants/tools/layer_scan_imports.py . "$S/imps.json"
for rule in L1 L2 L3; do
  for ctx in module tc func; do
    python3 /mnt/project-files/invariants/tools/layer_scan_rules.py "$S/imps.json" "$rule" "$ctx"
  done
done
python3 /mnt/project-files/architecture-docs/tools/layer_matrix.py "$S/imps.json"
```

Add `v` as a fourth argument to `layer_scan_rules.py` to list each site

## Proxy modules loaded by `import litellm`

This needs a venv with only the SDK installed (no proxy extras)

```bash
uv venv "$S/sdk" && VIRTUAL_ENV="$S/sdk" uv pip install -e .
"$S/sdk/bin/python" -c "import sys, time; t=time.time(); import litellm; \
print(round(time.time()-t,1), 's', len([m for m in sys.modules if m.startswith('litellm')]), 'modules', \
sorted(m for m in sys.modules if m.startswith('litellm.proxy')))"
```

## Service-locator reads of `proxy_server` globals (page 02)

```bash
rg -n 'from litellm.proxy.proxy_server import' litellm --type py -g '!litellm/proxy/proxy_server.py' | wc -l
rg -l 'from litellm.proxy.proxy_server import' litellm --type py | wc -l
rg -oN 'from litellm.proxy.proxy_server import \(?[a-z_, ]+' litellm --type py \
  | sed 's/.*import (\?//' | tr ',' '\n' | tr -d ' ' | rg -v '^$' | sort | uniq -c | sort -rn | head
```

## File sizes (gap G-L8)

```bash
wc -l litellm/proxy/proxy_server.py litellm/router.py litellm/main.py litellm/utils.py \
  litellm/llms/custom_httpx/llm_http_handler.py litellm/litellm_core_utils/litellm_logging.py | sort -n
```

## Rust rollout (page 06)

```bash
sed -n '/^RULES/,/^)/p' litellm/rust_bridge/catalog.py
```

## Lazy features and the componentized allowlists (page 07)

```bash
rg -c 'LazyFeature\(' litellm/proxy/_lazy_features.py
uv run pytest tests/unit/proxy/test_component_allowlists.py -q
```

## Background job defaults (control-plane flows)

```bash
rg -n 'PROXY_BATCH_WRITE_AT|PROXY_CONFIG_RELOAD_INTERVAL_SECONDS|PROXY_BUDGET_RESCHEDULER' litellm/constants.py
```

When a number changes, update it on the page, update the "Verified on" line to the new commit, and check whether the change closes or opens a row in [gap-register.md](../reference/gap-register.md)

## Redraw the diagrams

Every picture in the series is an SVG in `diagrams/`, drawn by three scripts in `tools/diagrams/` that only need the Python standard library. `build_explanation.py` draws the flowcharts on the explanation pages, `build_sequences.py` draws the user-flow and hook sequence diagrams, and `build_reference.py` draws the surface map and reads the flow and subsystem grid straight from the table in `reference/flow-subsystem-matrix.md`. Flowcharts are laid out by hand with explicit coordinates, so moving a box means editing its numbers. Sequence diagrams lay themselves out from a list of steps, so adding a message is one line

```bash
cd /mnt/project-files/architecture-docs/tools/diagrams
python3 build_explanation.py && python3 build_sequences.py && python3 build_reference.py
```

Open the changed SVG in a browser to check that labels don't overlap before committing to the new picture
