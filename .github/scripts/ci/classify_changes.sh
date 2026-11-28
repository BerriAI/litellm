#!/usr/bin/env bash
set -uo pipefail

category="${1:?usage: classify_changes.sh <backend|client|ui|cost-map-only|windows-release|redis-compat|python-310|helm|dashboard-build>}"

has_client=false
has_backend=false
has_ci=false
has_cost_map=false
has_windows_release=false
has_redis_compat=false
has_python_310=false
has_helm=false
has_dashboard_build=false
outside_cost_map_set=false
while IFS= read -r file || [ -n "$file" ]; do
  [ -n "$file" ] || continue
  case "$file" in
    litellm/_redis.py | litellm/_redis_credential_provider.py | litellm/caching/redis_cache.py | litellm/caching/evicted_client_closer.py | tests/unit/test_redis.py | tests/integration/sdk/test_redis_cluster_iam_auth.py | tests/unit/caching/test_redis_connection_pool.py | tests/unit/caching/test_redis_cluster_cache.py | tests/unit/caching/test_evicted_client_closer.py | .github/workflows/test-unit.yml | .github/actions/detect-changes/* | .github/scripts/detect_changes.sh | .github/scripts/ci/classify_changes.sh | tests/integration/run.py | pyproject.toml | uv.lock)
      has_redis_compat=true ;;
  esac
  case "$file" in
    litellm/proxy/db/prisma_engine.py | litellm/proxy/db/prisma_client.py | tests/unit/proxy/db/test_prisma_engine.py | tests/unit/proxy/db/test_prisma_planned_engine_restart.py | tests/unit/proxy/db/conftest.py | tests/_process_helpers.py | litellm/integrations/custom_logger_v2/*.py | tests/unit/integrations/custom_logger_v2/test_dispatcher.py | .github/workflows/test-unit.yml | .github/actions/detect-changes/* | .github/scripts/detect_changes.sh | .github/scripts/ci/classify_changes.sh | pyproject.toml | uv.lock)
      has_python_310=true ;;
  esac
  case "$file" in
    litellm/rust/* | litellm/rust_bridge/* | pyproject.toml | uv.lock | tests/windows_tests/* | .github/scripts/ci/* | .github/actions/*)
      has_windows_release=true ;;
  esac
  case "$file" in
    litellm/deploy/helm/*) has_helm=true ;;
    litellm/ui/litellm-dashboard/* | litellm/deploy/docker/Dockerfile | litellm/deploy/docker/Dockerfile.dockerignore) has_dashboard_build=true ;;
  esac
  case "$file" in
    litellm/ui/* | tests/e2e/ui/*) has_client=true ;;
    *.md | *.mdx) : ;;
    .github/*) has_ci=true; has_backend=true ;;
    *) has_backend=true ;;
  esac
  case "$file" in
    model_prices_and_context_window.json | litellm/model_prices_and_context_window_backup.json)
      has_cost_map=true ;;
    tests/unit/proxy/*) : ;;
    *) outside_cost_map_set=true ;;
  esac
done

case "$category" in
  cost-map-only)
    { [ "$has_cost_map" = true ] && [ "$outside_cost_map_set" = false ]; } && echo run || echo skip
    ;;
  windows-release)
    [ "$has_windows_release" = true ] && echo run || echo skip
    ;;
  redis-compat)
    [ "$has_redis_compat" = true ] && echo run || echo skip
    ;;
  python-310)
    [ "$has_python_310" = true ] && echo run || echo skip
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
  helm)
    { [ "$has_helm" = true ] || [ "$has_ci" = true ]; } && echo run || echo skip
    ;;
  dashboard-build)
    { [ "$has_dashboard_build" = true ] || [ "$has_ci" = true ]; } && echo run || echo skip
    ;;
  *)
    echo run
    ;;
esac
