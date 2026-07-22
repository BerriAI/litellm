#!/usr/bin/env bash
# Regression net for the prisma bake in the shipped images.
#
# Boots a freshly built image's migration entrypoint the way an OpenShift /
# air-gapped deployment does — an internal-only network (no egress) and an
# arbitrary non-root uid in GID 0 — against a brand-new Postgres, then asserts
# the schema actually got created.
#
# This catches the whole failure class, not one symptom: a bake that only works
# under `docker run` as the default uid with network still passes every existing
# check, because the migration entrypoint exits 0 even when it applied nothing.
# Asserting the table count is what turns that silent success into a hard fail.
#
# Usage: docker/test_offline_migration.sh <image-tag>
set -euo pipefail

IMAGE="${1:?usage: test_offline_migration.sh <image-tag>}"
PG_IMAGE="postgres:16-alpine"
MIN_TABLES="${MIN_TABLES:-20}"
PG_READY_ATTEMPTS="${PG_READY_ATTEMPTS:-60}"
RUN_ID="offlinemig-$$"
NET="${RUN_ID}-net"
PG="${RUN_ID}-pg"
PG_PW="pw"
PG_DB="litellm"
NON_ROOT_UID="12345:0"

log() { echo "[offline-migration] $*"; }

dump_diagnostics() {
  log "diagnostics:"
  docker ps -a --filter "name=${PG}" || true
  log "postgres logs:"
  docker logs "$PG" 2>&1 | tail -30 || true
}

cleanup() {
  docker rm -f "$PG" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Pre-pull Postgres while egress still exists; the internal network below has none.
log "pulling ${PG_IMAGE}"
docker pull -q "$PG_IMAGE" >/dev/null

# --internal => the litellm container cannot reach binaries.prisma.sh or npm.
log "creating internal network ${NET}"
docker network create --internal "$NET" >/dev/null

log "starting postgres"
docker run -d --name "$PG" --network "$NET" \
  -e POSTGRES_PASSWORD="$PG_PW" -e POSTGRES_DB="$PG_DB" "$PG_IMAGE" >/dev/null

log "waiting for postgres (up to ${PG_READY_ATTEMPTS}s)"
ready=""
for attempt in $(seq 1 "$PG_READY_ATTEMPTS"); do
  if ! docker ps --filter "name=${PG}" --filter "status=running" --format '{{.Names}}' | grep -q "$PG"; then
    log "postgres container is not running (attempt ${attempt})"
    dump_diagnostics
    exit 1
  fi
  if docker exec "$PG" pg_isready -U postgres -d "$PG_DB" >/dev/null 2>&1; then
    ready="yes"
    log "postgres ready after ${attempt}s"
    break
  fi
  sleep 1
done

if [ -z "$ready" ]; then
  log "postgres never became ready after ${PG_READY_ATTEMPTS}s"
  dump_diagnostics
  exit 1
fi

log "running migration as uid ${NON_ROOT_UID} on an internal-only network (no egress)"
set +e
docker run --rm --network "$NET" --user "$NON_ROOT_UID" \
  -e DATABASE_URL="postgresql://postgres:${PG_PW}@${PG}:5432/${PG_DB}" \
  -e LITELLM_MASTER_KEY="sk-offline-migration-test" \
  -e DISABLE_SCHEMA_UPDATE="false" \
  -w /app --entrypoint python \
  "$IMAGE" litellm/proxy/prisma_migration.py
migrate_rc=$?
set -e

tables=$(docker exec "$PG" psql -U postgres -d "$PG_DB" -tAc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d '[:space:]')

echo "----"
log "migration exit code: ${migrate_rc}"
log "public tables created: ${tables:-0} (require >= ${MIN_TABLES})"

if [ "${tables:-0}" -lt "$MIN_TABLES" ]; then
  log "FAIL: schema was not created offline as a non-root uid."
  log "      The image's prisma bake is not self-contained (it needs a runtime"
  log "      download or a writable HOME/cache), so OpenShift and air-gapped"
  log "      deployments start on an empty database and every DB endpoint 500s."
  exit 1
fi

log "PASS: full schema migrated offline as a non-root uid."
