#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for count in 2 3 4 5; do
  curl --silent --show-error --dump-header "register-${count}.headers.txt" --output "register-${count}.response.json" --write-out "${count} redirect URIs: HTTP %{http_code}\n" -X POST http://127.0.0.1:47449/register -H 'Content-Type: application/json' --data-binary "@register-${count}.request.json"
done
