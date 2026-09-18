#!/usr/bin/env bash
set -uo pipefail

category="${1:?usage: classify_changes.sh <backend|client|ui|provider-harness|cost-map-only>}"

has_client=false
has_backend=false
has_ci=false
has_provider_harness=false
has_cost_map=false
outside_cost_map_set=false
while IFS= read -r file || [ -n "$file" ]; do
  [ -n "$file" ] || continue
  case "$file" in
    tests/e2e/*/*.py) : ;;
    tests/e2e/*.py | tests/code_coverage_tests/test_provider_cache.py | tests/code_coverage_tests/test_provider_replay_harness.py | tests/test_litellm/test_circleci_path_filter.py | .circleci/* | pyproject.toml | uv.lock)
      has_provider_harness=true ;;
  esac
  case "$file" in
    ui/* | tests/e2e/ui/*) has_client=true ;;
    docs/* | *.md | *.mdx) : ;;
    .github/* | .circleci/*) has_ci=true; has_backend=true ;;
    *) has_backend=true ;;
  esac
  case "$file" in
    model_prices_and_context_window.json | litellm/model_prices_and_context_window_backup.json | model_prices_and_context_window.schema.json)
      has_cost_map=true ;;
    tests/test_litellm/* | tests/proxy_unit_tests/*) : ;;
    *) outside_cost_map_set=true ;;
  esac
done

case "$category" in
  cost-map-only)
    { [ "$has_cost_map" = true ] && [ "$outside_cost_map_set" = false ]; } && echo run || echo skip
    ;;
  provider-harness)
    [ "$has_provider_harness" = true ] && echo run || echo skip
    ;;
  backend)
    [ "$has_backend" = true ] && echo run || echo skip
    ;;
  client)
    { [ "$has_client" = true ] || [ "$has_backend" = true ]; } && echo run || echo skip
    ;;
  ui)
    { [ "$has_client" = true ] || [ "$has_ci" = true ]; } && echo run || echo skip
    ;;
  *)
    echo run
    ;;
esac
