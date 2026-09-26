#!/usr/bin/env bash
# Refresh the sqlx cache and run every check a shared query must pass. Needs Docker and sqlx-cli.
set -euo pipefail

command -v cargo-sqlx >/dev/null || {
  echo "cargo-sqlx not found: cargo install sqlx-cli --version 0.9.0 --locked --no-default-features --features rustls,postgres" >&2
  exit 1
}

cd "$(git rev-parse --show-toplevel)"

echo "== refresh litellm-rust/crates/db/.sqlx"
make rust-sqlx-prepare

echo "== clippy, offline"
(cd litellm-rust && env -u DATABASE_URL cargo clippy -p litellm-db --all-targets --locked --features postgres-tests,schema -- -D warnings)

echo "== postgres-tests"
(cd litellm-rust && cargo test -p litellm-db --locked --features postgres-tests)

echo "== shared query gate"
uv run --no-sync python tests/code_coverage_tests/check_shared_sql_queries.py

echo "all checks passed; commit litellm-rust/crates/db/.sqlx with the change"
