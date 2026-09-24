#!/usr/bin/env bash
set -euo pipefail

flag="${1:?usage: unit_selection.sh <codecov flag>}"

legacy_flags=(
  caching-local
  enterprise-package
  enterprise-routing
  proxy-extras
  proxy-infra
)

legacy_paths() {
  case "$1" in
    caching-local) echo tests/unit/caching ;;
    enterprise-package)
      echo tests/unit/enterprise/integrations
      echo tests/unit/enterprise/proxy/auth
      echo tests/unit/enterprise/proxy/guardrails
      echo tests/unit/enterprise/proxy/hooks
      echo tests/unit/enterprise/proxy/management_endpoints
      echo tests/unit/enterprise/proxy/test_audit_logging_endpoints.py
      echo tests/unit/enterprise/enterprise_callbacks/test_prometheus_logging_callbacks.py ;;
    enterprise-routing)
      echo tests/unit/enterprise/enterprise_callbacks/send_emails
      echo tests/unit/enterprise/proxy/test_afile_retrieve_returns_unified_id.py
      echo tests/unit/enterprise/proxy/test_batch_retrieve_input_file_id.py
      echo tests/unit/enterprise/proxy/test_batch_retrieve_registers_missing_output_file_id.py
      echo tests/unit/enterprise/proxy/test_batch_retrieve_returns_unified_input_file_id.py
      echo tests/unit/enterprise/proxy/test_batch_update_db_managed_output_file_id.py
      echo tests/unit/enterprise/proxy/test_deleted_file_returns_403_not_404.py
      echo tests/unit/enterprise/proxy/test_enterprise_routes.py
      echo tests/unit/enterprise/proxy/test_file_deletion_blocking.py
      echo tests/unit/enterprise/proxy/test_managed_files_access_check.py
      echo tests/unit/enterprise/proxy/test_managed_files_hook.py ;;
    proxy-extras) echo tests/unit/litellm_proxy_extras ;;
    proxy-infra) echo tests/unit/gateway ;;
    *) echo "unit_selection.sh: unknown flag $1" >&2; exit 1 ;;
  esac
}

expand() {
  while read -r path; do
    if [ -d "$path" ]; then
      find "$path" -name 'test_*.py'
    elif [ -f "$path" ]; then
      echo "$path"
    else
      echo "unit_selection.sh: $path does not exist" >&2
      exit 1
    fi
  done
}

if [ "$flag" = unit ]; then
  comm -23 \
    <(find tests/unit -name 'test_*.py' | sort) \
    <(for legacy in "${legacy_flags[@]}"; do legacy_paths "$legacy"; done | expand | sort)
  exit 0
fi

legacy_paths "$flag" | expand | sort
