#!/usr/bin/env bash
set -uo pipefail

category="${1:?usage: classify_changes.sh <backend|client|ui|provider-harness|cost-map-only|mcp-dependencies|windows-release|redis-compat>}"

has_client=false
has_backend=false
has_ci=false
has_provider_harness=false
has_cost_map=false
has_mcp_dependencies=false
has_windows_release=false
has_redis_compat=false
outside_cost_map_set=false
while IFS= read -r file || [ -n "$file" ]; do
  [ -n "$file" ] || continue
  case "$file" in
    litellm/_redis.py | litellm/_redis_credential_provider.py | litellm/caching/redis_cache.py | litellm/caching/evicted_client_closer.py | tests/unit/test_redis.py | tests/local_testing/test_caching.py | tests/unit/caching/test_redis_connection_pool.py | tests/unit/caching/test_redis_cluster_cache.py | tests/unit/caching/test_evicted_client_closer.py | .circleci/config.yml | .circleci/scripts/classify_changes.sh | .circleci/scripts/path_filter.sh | pyproject.toml | uv.lock)
      has_redis_compat=true ;;
  esac
  case "$file" in
    *.md | *.mdx) : ;;
    pyproject.toml | */pyproject.toml | uv.lock | uv.toml | .python-version | rust-toolchain.toml | litellm-rust/* | litellm/__init__.py | litellm/proxy/proxy_server.py | litellm/*mcp* | tests/*mcp* | litellm/integrations/arize/* | tests/base_sdk_tests/* | scripts/check_mcp_sdk_install.py | .github/workflows/test-mcp-dependency-resolution.yml | .github/actions/detect-changes/* | .github/actions/setup-uv-with-retries/* | .github/actions/cache-cargo-build/* | .github/scripts/detect_changes.sh | .github/scripts/uv_sync_with_retries.sh | .circleci/scripts/classify_changes.sh | tests/unit/test_circleci_path_filter.py | tests/unit/test_detect_changes.py)
      has_mcp_dependencies=true ;;
  esac
  case "$file" in
    tests/e2e/*/*.py | tests/e2e_harness/*/*.py) : ;;
    tests/e2e/*.py | tests/e2e_harness/*.py | tests/code_coverage_tests/test_provider_cache.py | tests/code_coverage_tests/test_provider_replay_harness.py | tests/unit/test_circleci_path_filter.py | .circleci/* | pyproject.toml | uv.lock)
      has_provider_harness=true ;;
  esac
  case "$file" in
    litellm-rust/* | litellm/rust_bridge/* | rust-toolchain.toml | pyproject.toml | uv.lock | tests/windows_tests/* | .circleci/*)
      has_windows_release=true ;;
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
    tests/test_litellm/* | tests/proxy_unit_tests/* | tests/unit/proxy/*) : ;;
    *) outside_cost_map_set=true ;;
  esac
done

case "$category" in
  mcp-dependencies)
    [ "$has_mcp_dependencies" = true ] && echo run || echo skip
    ;;
  cost-map-only)
    { [ "$has_cost_map" = true ] && [ "$outside_cost_map_set" = false ]; } && echo run || echo skip
    ;;
  provider-harness)
    [ "$has_provider_harness" = true ] && echo run || echo skip
    ;;
  windows-release)
    [ "$has_windows_release" = true ] && echo run || echo skip
    ;;
  redis-compat)
    [ "$has_redis_compat" = true ] && echo run || echo skip
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
