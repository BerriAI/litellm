#!/bin/sh
# Fails the FIPS image build unless every ELF under the given roots is either crypto-free and listed in
# fips_elf_inventory.txt, linked to the system libcrypto, or a waived binary that bundles its own crypto.
# Regenerate the inventory after a reviewed dependency change with:
#   docker run --rm -v "$PWD/litellm/deploy/docker:/scan" --entrypoint sh <image> -c \
#     'SCAN_EMIT_INVENTORY=1 /scan/scan_bundled_crypto.sh /app/.venv /opt/prisma /usr/local/bin' \
#     > litellm/deploy/docker/fips_elf_inventory.txt
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
inventory="${SCAN_INVENTORY:-$script_dir/fips_elf_inventory.txt}"
report="${SCAN_REPORT:-/fips-scan-report.txt}"
failures="$(mktemp)"
counts="$(mktemp)"
files="$(mktemp)"
trap 'rm -f "$failures" "$counts" "$files"' EXIT

normalize() {
  printf '%s\n' "$1" | sed \
    -e 's/aarch64/ARCH/g' \
    -e 's/x86_64/ARCH/g' \
    -e 's/-[0-9a-f]\{8,\}\(\.so\)/\1/'
}

is_elf() {
  [ "$(head -c 4 "$1" 2>/dev/null)" = "$(printf '\177ELF')" ]
}

crypto_markers() {
  m=""
  if grep -aq 'BoringSSL' "$1"; then m="BoringSSL"; fi
  if grep -aq 'AWS-LC' "$1"; then m="${m:+$m,}AWS-LC"; fi
  if grep -Eaq 'OpenSSL [0-9]+\.[0-9]+\.[0-9]+' "$1"; then
    if ! objdump -p "$1" 2>/dev/null | grep -Eq 'NEEDED.*libcrypto\.so\.[34]'; then
      m="${m:+$m,}bundled-OpenSSL"
    fi
  fi
  if grep -aq 'sodium_init' "$1"; then m="${m:+$m,}libsodium"; fi
  if grep -aq 'ring_core_' "$1"; then m="${m:+$m,}ring"; fi
  if grep -aq 'rustls' "$1"; then m="${m:+$m,}rustls"; fi
  if grep -Eaqi 'mbed.?tls' "$1"; then m="${m:+$m,}mbedTLS"; fi
  if grep -aq 'wolfSSL' "$1"; then m="${m:+$m,}wolfSSL"; fi
  if grep -aq 'libgcrypt' "$1"; then m="${m:+$m,}libgcrypt"; fi
  printf '%s' "$m"
}

is_waived() {
  case "$1" in
    */site-packages/grpc/_cython/cygrpc.cpython-*.so) ;;
    */site-packages/hf_xet/hf_xet.abi3.so) ;;
    */site-packages/_awscrt.abi3.so) ;;
    */site-packages/pyroscope/_native__lib.cpython-*.so) ;;
    */site-packages/psycopg_binary.libs/libcrypto-*.so.*) ;;
    */site-packages/nacl/_sodium.abi3.so) ;;
    */site-packages/litellm/rust_bridge/_native.abi3.so) ;;
    */site-packages/granian/_granian.cpython-*.so) ;;
    */site-packages/_polars_runtime_32/_polars_runtime.abi3.so) ;;
    */site-packages/ddtrace/internal/native/_native.cpython-*.so) ;;
    /opt/prisma/*/query-engine*|/opt/prisma/*/schema-engine*) ;;
    *) return 1 ;;
  esac
}

if [ "$#" -eq 0 ]; then
  echo "FAIL: no scan roots given" >&2
  exit 1
fi
for root in "$@"; do
  if [ ! -e "$root" ]; then
    echo "FAIL: scan root $root does not exist" >&2
    exit 1
  fi
done
# Not piped, so set -e stops the scan when find cannot read a directory
find "$@" -type f > "$files"

if [ "${SCAN_EMIT_INVENTORY:-}" = "1" ]; then
  while read -r f; do
    if is_elf "$f"; then
      normalize "$f"
    fi
  done < "$files" | sort -u
  exit 0
fi

if [ ! -f "$inventory" ]; then
  echo "FAIL: inventory file not found at $inventory" >&2
  exit 1
fi

: > "$report"
sort "$files" | while read -r f; do
  if ! is_elf "$f"; then
    continue
  fi
  echo scanned >> "$counts"
  norm="$(normalize "$f")"
  markers="$(crypto_markers "$f")"
  if [ -n "$markers" ]; then
    printf 'BUNDLED-CRYPTO %s [%s]\n' "$f" "$markers" >> "$report"
    if is_waived "$f"; then
      printf 'WAIVED    %s [%s]\n' "$f" "$markers"
    else
      printf 'VIOLATION %s [%s] bundled crypto with no waiver\n' "$f" "$markers"
      echo "$f" >> "$failures"
    fi
  elif grep -Fxq "$norm" "$inventory"; then
    printf 'CLEAN %s\n' "$f" >> "$report"
  else
    printf 'VIOLATION %s [unreviewed] not in fips_elf_inventory.txt; review it, then regenerate the inventory\n' "$f"
    echo "$f" >> "$failures"
  fi
done

scanned="$(grep -c scanned "$counts" || true)"
if [ -s "$failures" ]; then
  printf 'FAIL: %s of %s ELF binaries unaccounted for (see above)\n' "$(wc -l < "$failures" | tr -d ' ')" "$scanned"
  exit 1
fi
if [ "$scanned" -eq 0 ]; then
  printf 'FAIL: no ELF binaries found under %s\n' "$*"
  exit 1
fi
printf 'scanned %s ELF binaries\n' "$scanned" >> "$report"
printf 'PASS: all %s ELF binaries accounted for (inventoried, system-linked, or waived)\n' "$scanned"
