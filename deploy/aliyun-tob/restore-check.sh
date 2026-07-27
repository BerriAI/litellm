#!/bin/sh
set -eu

dump_file=$1
test -f "${dump_file}"
docker compose --env-file /opt/litellm-relay/.env -f /opt/litellm-relay/docker-compose.yml exec -T database \
  dropdb -U litellm --if-exists litellm_restore_check
docker compose --env-file /opt/litellm-relay/.env -f /opt/litellm-relay/docker-compose.yml exec -T database \
  createdb -U litellm litellm_restore_check
docker compose --env-file /opt/litellm-relay/.env -f /opt/litellm-relay/docker-compose.yml exec -T database \
  pg_restore -U litellm -d litellm_restore_check --no-owner < "${dump_file}"
docker compose --env-file /opt/litellm-relay/.env -f /opt/litellm-relay/docker-compose.yml exec -T database \
  psql -U litellm -d litellm_restore_check -c 'SELECT COUNT(*) FROM "LiteLLM_PublicRelayAccount"'
docker compose --env-file /opt/litellm-relay/.env -f /opt/litellm-relay/docker-compose.yml exec -T database \
  dropdb -U litellm litellm_restore_check
