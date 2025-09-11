#!/bin/bash
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euox pipefail
IFS=$'\n\t'

usage() {
    echo "Usage :"
    echo "$0 <version> <path>"
    echo ""
    echo "Example"
    echo "  $0 v0.7.0-rc.1 ./vendor"
}

if [ $# != 2 ] || [ "$1" == "-h" ]; then
    usage
    exit 1
fi

SCRIPTPATH=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPTPATH")

MARCH=$(uname -m)

TAG_INFER=$1
TARGET_EXTRACT=$2

CHECKSUM_FILE=${SCRIPTDIR}/infer_checksums.txt

# https://github.com/facebook/infer/releases/download/v1.1.0/infer-linux64-v1.1.0.tar.xz
TAR_INFER=infer-linux64-${TAG_INFER}.tar.xz
GITHUB_URL_INFER=https://github.com/facebook/infer/releases/download/${TAG_INFER}/${TAR_INFER}

SHA256_INFER="blank"
while IFS='  ' read -r checksum filename; do
    if [ "$filename" == "$TAR_INFER" ]; then
        SHA256_INFER="$checksum  $filename"
        break
    fi
done < "$CHECKSUM_FILE"

if [ "$SHA256_INFER" == "blank" ]; then
    echo "Could not find checksum for ${TAR_INFER} in ${CHECKSUM_FILE}"
    exit 1
else
  echo "Using infer sha256: ${SHA256_INFER}"
fi

mkdir -p "$TARGET_EXTRACT" || true
cd "$TARGET_EXTRACT"

if [[ -e "${TAR_INFER}" ]]; then
    already_present=1
else
    already_present=0
    echo "Downloading infer ${GITHUB_URL_INFER}..."
    if command -v curl > /dev/null 2>&1; then
        curl -fsSLO "${GITHUB_URL_INFER}"
    elif command -v wget > /dev/null 2>&1; then
        wget -q -O "${GITHUB_URL_INFER##*/}" "${GITHUB_URL_INFER}"
    else
        echo "Error: neither curl nor wget is available." >&2
        exit 1
    fi
fi

echo "Checking infer sha256"
if ! echo "${SHA256_INFER}" | sha256sum -c -; then
    echo "Error validating infer SHA256"
    echo "Please clear $TARGET_EXTRACT before restarting"
    exit 1
fi

if [[ $already_present -eq 0 ]]; then
    echo "Extracting ${TAR_INFER}"
    tar xf "${TAR_INFER}" --strip-components=1 --no-same-owner
fi
