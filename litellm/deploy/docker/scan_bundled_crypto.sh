#!/bin/sh
# Fails the FIPS image build unless every ELF under the given roots is either crypto-free and listed in
# fips_elf_inventory.txt, linked to the system libcrypto, or a waived binary that bundles its own crypto.
# A symlink is scanned at its own path when it points outside the roots, and fails the scan when it dangles or
# points at a folder outside them.
# Regenerate the inventory after a reviewed dependency change with:
#   docker run --rm -v "$PWD/litellm/deploy/docker:/scan" --entrypoint sh <image> -c \
#     'SCAN_EMIT_INVENTORY=1 /scan/scan_bundled_crypto.sh /app/.venv /opt/prisma /usr/local/bin' \
#     > litellm/deploy/docker/fips_elf_inventory.txt
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
inventory="${SCAN_INVENTORY:-$script_dir/fips_elf_inventory.txt}"
report="${SCAN_REPORT:-/fips-scan-report.txt}"
site_packages="${SCAN_SITE_PACKAGES:-/app/.venv/lib/python3.13/site-packages}"
prisma_root="${SCAN_PRISMA_ROOT:-/opt/prisma}"
failures="$(mktemp)"
counts="$(mktemp)"
files="$(mktemp)"
links="$(mktemp)"
trap 'rm -f "$failures" "$counts" "$files" "$links"' EXIT
nl='
'

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
  case "${1##*/}" in
    libssl[.-]*|libcrypto[.-]*|libcrypt[.-]*|libk5crypto[.-]*|libgnutls[.-]*|libnettle[.-]*|libhogweed[.-]*|\
    libgcrypt[.-]*|libsodium[.-]*|libmbedtls[.-]*|libmbedcrypto[.-]*|libmbedx509[.-]*|libwolfssl[.-]*|\
    libnss3[.-]*|libsoftokn3[.-]*|libfreebl3[.-]*|libfreeblpriv3[.-]*|libtomcrypt[.-]*|libbotan-*)
      m="${m:+$m,}crypto-library" ;;
  esac
  printf '%s' "$m"
}

# Each waiver is one exact path, after the same arch and hash normalization as the inventory
is_waived() {
  case "$1" in
    "$site_packages"/*) is_waived_package "$(normalize "${1#"$site_packages"/}")" ;;
    "$prisma_root"/*) is_waived_prisma "${1#"$prisma_root"/}" ;;
    *) return 1 ;;
  esac
}

is_waived_package() {
  case "$1" in
    hf_xet/hf_xet.abi3.so) ;;
    _awscrt.abi3.so) ;;
    pyroscope/_native__lib.cpython-39-ARCH-linux-gnu.so) ;;
    nacl/_sodium.abi3.so) ;;
    litellm/rust_bridge/_native.abi3.so) ;;
    granian/_granian.cpython-313-ARCH-linux-gnu.so) ;;
    _polars_runtime_32/_polars_runtime.abi3.so) ;;
    ddtrace/internal/native/_native.cpython-313-ARCH-linux-gnu.so) ;;
    *) return 1 ;;
  esac
}

is_waived_prisma() {
  case "$1" in
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/debian-openssl-1.1.x/query-engine) ;;
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/debian-openssl-3.0.x/query-engine) ;;
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/debian-openssl-3.0.x/schema-engine) ;;
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/linux-arm64-openssl-3.0.x/query-engine) ;;
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/linux-arm64-openssl-3.0.x/schema-engine) ;;
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/linux-musl-openssl-3.0.x/query-engine) ;;
    .cache/prisma/master/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/linux-musl/query-engine) ;;
    binaries/node_modules/@prisma/engines/query-engine-debian-openssl-3.0.x) ;;
    binaries/node_modules/@prisma/engines/query-engine-linux-arm64-openssl-3.0.x) ;;
    binaries/node_modules/@prisma/engines/schema-engine-debian-openssl-3.0.x) ;;
    binaries/node_modules/@prisma/engines/schema-engine-linux-arm64-openssl-3.0.x) ;;
    binaries/node_modules/prisma/query-engine-debian-openssl-1.1.x) ;;
    binaries/node_modules/prisma/query-engine-debian-openssl-3.0.x) ;;
    binaries/node_modules/prisma/query-engine-linux-arm64-openssl-3.0.x) ;;
    binaries/node_modules/prisma/query-engine-linux-musl) ;;
    binaries/node_modules/prisma/query-engine-linux-musl-openssl-3.0.x) ;;
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
find "$@" -type l > "$links"

# The file lists hold one path per line, so a newline in a name would split it in two
odd="$(find "$@" -name "*$nl*")"
if [ -n "$odd" ]; then
  printf 'FAIL: a path under the scan roots has a newline in its name: %s\n' "$odd" >&2
  exit 1
fi

roots=""
for root in "$@"; do
  roots="$roots$(readlink -f "$root")$nl"
done
inside_roots() {
  printf '%s' "$roots" | {
    while IFS= read -r r; do
      case "$1" in "$r" | "$r"/*) exit 0 ;; esac
    done
    exit 1
  }
}

while IFS= read -r l; do
  if [ ! -e "$l" ]; then
    printf 'FAIL: symlink %s points at nothing\n' "$l" >&2
    exit 1
  fi
  target="$(readlink -f "$l")"
  if inside_roots "$target"; then
    continue
  fi
  if [ -d "$l" ]; then
    printf 'FAIL: symlink %s points at a folder outside the scan roots: %s\n' "$l" "$target" >&2
    exit 1
  fi
  printf '%s\n' "$l" >> "$files"
done < "$links"

if [ "${SCAN_EMIT_INVENTORY:-}" = "1" ]; then
  while IFS= read -r f; do
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
sort "$files" | while IFS= read -r f; do
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
