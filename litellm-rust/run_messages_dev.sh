#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <rust|rust-bridge|legacy> --port <port>" >&2
}

mode="${1:-}"
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "$mode" ]]; then
  usage
  exit 2
fi
shift

port=""
while (($# > 0)); do
  case "$1" in
    --port)
      if (($# < 2)); then
        usage
        exit 2
      fi
      port="$2"
      shift 2
      ;;
    --port=*)
      port="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$mode" in
  rust | rust-bridge | legacy) ;;
  *)
    echo "unknown mode: $mode" >&2
    usage
    exit 2
    ;;
esac

if [[ ! "$port" =~ ^[0-9]+$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
  echo "--port must be an integer between 1 and 65535" >&2
  exit 2
fi

for name in AWS_BEARER_TOKEN_BEDROCK BEDROCK_MODEL LITELLM_MASTER_KEY; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required" >&2
    exit 1
  fi
done

if [[ ! "$BEDROCK_MODEL" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
  echo "BEDROCK_MODEL contains unsupported characters" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$script_dir")"
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-west-2}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

echo "starting $mode on http://127.0.0.1:$port with model $BEDROCK_MODEL"

if [[ "$mode" == "rust" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is required for rust mode" >&2
    exit 1
  fi
  cd "$script_dir"
  PORT="$port" REALTIME_POOL_SIZE=0 exec cargo run -p litellm-ai-gateway --features server
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for $mode mode" >&2
  exit 1
fi

cd "$repo_root"
if [[ "$mode" == "rust-bridge" ]]; then
  uv run --with maturin==1.9.4 maturin develop \
    --manifest-path "$script_dir/crates/python-bridge/Cargo.toml"
  export LITELLM_RUST=1
else
  export LITELLM_RUST=0
fi

dev_runtime_dir="$(mktemp -d)"
proxy_config="$dev_runtime_dir/config.yaml"
cleanup() {
  rm -f -- "$proxy_config"
  rmdir -- "$dev_runtime_dir" 2>/dev/null || true
}
trap cleanup EXIT

cat >"$proxy_config" <<EOF
model_list:
  - model_name: "$BEDROCK_MODEL"
    litellm_params:
      model: "bedrock/$BEDROCK_MODEL"
      api_key: os.environ/AWS_BEARER_TOKEN_BEDROCK
      aws_region_name: "$AWS_REGION_NAME"
EOF

uv run litellm --config "$proxy_config" --port "$port" --run_hypercorn
