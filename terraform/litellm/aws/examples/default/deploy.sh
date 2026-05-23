#!/usr/bin/env bash
#
# One-command LiteLLM deploy helper for AWS CloudShell (or any machine with
# the AWS CLI installed and credentials configured).
#
# AWS CloudShell ships git + the AWS CLI but not Terraform, so this script
# installs a pinned, checksum-verified Terraform into ./.bin if it isn't
# already on PATH, then runs `terraform init` + `terraform apply` against the
# trial root in this directory.
#
# Usage:
#   ./deploy.sh            # interactive apply (review plan, type yes)
#   AUTO_APPROVE=1 ./deploy.sh   # non-interactive
#
# Override the trial defaults with TF_VAR_* env vars or a terraform.tfvars
# file, e.g.:
#   export TF_VAR_region=us-east-1
#   ./deploy.sh
set -euo pipefail

TF_VERSION="1.9.8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Resolve a terraform binary (install a pinned one if missing) ----
if command -v terraform >/dev/null 2>&1; then
  TF="terraform"
else
  case "$(uname -m)" in
  x86_64 | amd64) ARCH="amd64" ;;
  aarch64 | arm64) ARCH="arm64" ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
  esac

  BIN_DIR="$SCRIPT_DIR/.bin"
  TF="$BIN_DIR/terraform"
  if [ ! -x "$TF" ]; then
    echo "Installing Terraform ${TF_VERSION} (${ARCH}) into ${BIN_DIR} ..."
    mkdir -p "$BIN_DIR"
    tmp="$(mktemp -d)"
    base="https://releases.hashicorp.com/terraform/${TF_VERSION}"
    zip="terraform_${TF_VERSION}_linux_${ARCH}.zip"
    # Download the artifact and HashiCorp's official checksum sidecar, then
    # verify before unpacking — never trust an unverified download.
    curl -fsSL -o "$tmp/$zip" "$base/$zip"
    curl -fsSL -o "$tmp/SHA256SUMS" "$base/terraform_${TF_VERSION}_SHA256SUMS"
    (cd "$tmp" && grep " $zip\$" SHA256SUMS | sha256sum -c -)
    unzip -o "$tmp/$zip" -d "$BIN_DIR" >/dev/null
    rm -rf "$tmp"
  fi
fi

echo "Using $("$TF" version | head -1)"

# ---- Sanity-check AWS credentials before spending 15+ minutes applying ----
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS credentials not found. In CloudShell this is automatic; otherwise run 'aws configure' or set AWS_PROFILE." >&2
  exit 1
fi

# ---- Deploy ----
"$TF" init -input=false
if [ "${AUTO_APPROVE:-}" = "1" ]; then
  "$TF" apply -auto-approve -input=false
else
  "$TF" apply -input=false
fi

echo
echo "==================================================================="
echo " LiteLLM is deploying. Useful outputs:"
echo "==================================================================="
URL="$("$TF" output -raw alb_url 2>/dev/null || true)"
KEY_ARN="$("$TF" output -raw master_key_secret_arn 2>/dev/null || true)"
echo "  Proxy URL : ${URL:-<run: terraform output alb_url>}"
echo "  UI login  : admin / <master key>"
if [ -n "$KEY_ARN" ]; then
  echo "  Master key: aws secretsmanager get-secret-value --secret-id $KEY_ARN --query SecretString --output text"
fi
echo
echo "The ALB takes a few minutes to pass health checks after apply returns."
