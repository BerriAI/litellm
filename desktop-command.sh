#!/bin/bash
set -euo pipefail
docker exec -e DISPLAY=:99 litellm-7449-vscode sh -c 'wid=$(xdotool search --onlyvisible --class Code | head -1); xdotool windowactivate --sync "$wid"; xdotool key --clearmodifiers ctrl+shift+p'
sleep 0.5
docker exec -e DISPLAY=:99 litellm-7449-vscode xdotool type --clearmodifiers --delay 30 "$1"
sleep 0.5
docker exec -e DISPLAY=:99 litellm-7449-vscode xdotool key --clearmodifiers Return
