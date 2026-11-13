#!/usr/bin/env bash
set -euo pipefail

bench_root="$(cd "$(dirname "$0")" && pwd)"
tools_dir="${BENCH_TOOLS_DIR:-$bench_root/runtime/tools}"
mkdir -p "$tools_dir"

download() {
    local url="$1" destination="$2" checksum="$3"
    curl --fail --location "$url" --output "$destination"
    printf '%s  %s\n' "$checksum" "$destination" | shasum -a 256 --check -
}

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) go_target=darwin-arm64; go_sha=76812b213b1b2302c978d28fa52fa92d541704b9e7d9d5db8002c50e4018c4c5 ;;
    Darwin-x86_64) go_target=darwin-amd64; go_sha=587b59182488b23aa6e5fc25110405a3e0e5b38ed2f5b2f46ed13c32aee356fe ;;
    Linux-aarch64) go_target=linux-arm64; go_sha=94f3e30b8e374bc285e7dadc11e0865726b9bc6e85b841ccceaabc0214c6b7c8 ;;
    Linux-x86_64) go_target=linux-amd64; go_sha=ecbadb99091a3f46e31f5f934b068b1864eafa7995211b39eaddf76996045fe5 ;;
    *) printf 'Unsupported Go platform\n' >&2; exit 1 ;;
esac
download "https://go.dev/dl/go1.27.2.$go_target.tar.gz" "$tools_dir/go.tar.gz" "$go_sha"
tar -xzf "$tools_dir/go.tar.gz" -C "$tools_dir"
suite_commit=3cb72a95cacaee6f73f9db5636a1a651c7dcb1e9
if [ ! -d "$tools_dir/bifrost-benchmarking/.git" ]; then
    git clone https://github.com/maximhq/bifrost-benchmarking.git "$tools_dir/bifrost-benchmarking"
fi
git -C "$tools_dir/bifrost-benchmarking" checkout --detach "$suite_commit"
if git -C "$tools_dir/bifrost-benchmarking" apply --check "$bench_root/bifrost.patch"; then
    git -C "$tools_dir/bifrost-benchmarking" apply "$bench_root/bifrost.patch"
else
    git -C "$tools_dir/bifrost-benchmarking" apply --reverse --check "$bench_root/bifrost.patch"
fi
touch "$tools_dir/bifrost-benchmarking/.env"
(
    cd "$tools_dir/bifrost-benchmarking"
    CGO_ENABLED=0 "$tools_dir/go/bin/go" build -trimpath -mod=readonly -o "$tools_dir/suite" benchmark.go
)
(
    cd "$bench_root"
    cargo build --release --locked
)
