#!/bin/sh
# Moved to litellm/scripts/quickstart.sh. Old links run the current script with the same arguments.
set -eu
script="$(curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/scripts/quickstart.sh)"
exec sh -c "$script" quickstart.sh "$@"
