#!/bin/sh
set -eu
umask 077

backup_root=/opt/litellm-relay/backups
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
daily_file="${backup_root}/daily/litellm-${timestamp}.dump"
daily_tmp="${daily_file}.tmp"
weekly_file="${backup_root}/weekly/litellm-${timestamp}.dump"

mkdir -p "${backup_root}/daily" "${backup_root}/weekly"
trap 'rm -f "${daily_tmp}"' EXIT
docker compose --env-file /opt/litellm-relay/.env -f /opt/litellm-relay/docker-compose.yml exec -T database \
  pg_dump -U litellm -d litellm -Fc > "${daily_tmp}"
test -s "${daily_tmp}"
mv "${daily_tmp}" "${daily_file}"
find "${backup_root}/daily" -type f -name 'litellm-*.dump' -mtime +7 -delete

if [ "$(date -u +%u)" = "7" ]; then
  cp "${daily_file}" "${weekly_file}"
  find "${backup_root}/weekly" -type f -name 'litellm-*.dump' -mtime +28 -delete
fi
