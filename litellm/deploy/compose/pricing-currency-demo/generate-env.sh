#!/bin/sh
# Writes demo-only credentials to .env beside this script. An existing .env is kept, since the salt key in it
# encrypts what the demo database already stores.
set -eu
cd "$(dirname "$0")"
if [ -f .env ]; then
    echo ".env already exists, keeping it"
    exit 0
fi
umask 077
random() { openssl rand -hex "$1"; }
cat > .env <<ENV
LITELLM_MASTER_KEY=sk-$(random 24)
LITELLM_SALT_KEY=sk-$(random 24)
POSTGRES_PASSWORD=$(random 16)
UI_PASSWORD=$(random 12)
DEMO_VIRTUAL_KEY=sk-$(random 20)
DEMO_PRICING_KEY=sk-$(random 20)
ENV
echo "Wrote .env with demo-only keys. UI login: admin / UI_PASSWORD from .env"
