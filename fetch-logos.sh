#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/docs/assets/logos"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$OUT_DIR"

ICONS=(
  "anthropic|anthropic.svg"
  "azure|azure-color.svg"
  "bedrock|bedrock-color.svg"
  "claude|claude-color.svg"
  "cursor|cursor.svg"
  "mistral|mistral-color.svg"
  "notion|notion.svg"
  "openai|openai.svg"
  "vertexai|vertexai-color.svg"
)

successes=()
misses=()

fetch_icon() {
  local name="$1"
  local package_file="$2"
  local dest="$OUT_DIR/$name.svg"
  local tmp="$TMP_DIR/$name.svg"
  local urls=(
    "https://unpkg.com/@lobehub/icons-static-svg@latest/icons/$package_file"
    "https://registry.npmmirror.com/@lobehub/icons-static-svg/latest/files/icons/$package_file"
  )

  for url in "${urls[@]}"; do
    if curl -fsSL "$url" -o "$tmp" && grep -qi '<svg' "$tmp"; then
      cp "$tmp" "$dest"
      successes+=("$name.svg <= $url")
      return 0
    fi
  done

  rm -f "$tmp" "$dest"
  misses+=("$name.svg")
  return 1
}

for icon in "${ICONS[@]}"; do
  name="${icon%%|*}"
  package_file="${icon#*|}"
  fetch_icon "$name" "$package_file" || true
done

echo "Succeeded:"
if ((${#successes[@]})); then
  printf '  - %s\n' "${successes[@]}"
else
  echo "  - none"
fi

echo "Missed:"
if ((${#misses[@]})); then
  printf '  - %s\n' "${misses[@]}"
else
  echo "  - none"
fi

echo "Known package gaps:"
echo "  - codex.svg (using openai.svg in docs/assets/hero.html)"
echo "  - openclaw.svg (using emoji fallback in docs/assets/hero.html)"
