#!/bin/bash

# Check if nvm is not installed
if ! command -v nvm &> /dev/null; then
  # Install nvm with checksum verification
  NVM_VERSION="v0.40.4"
  NVM_CHECKSUM="4b7412c49960c7d31e8df72da90c1fb5b8cccb419ac99537b737028d497aba4f"
  NVM_SCRIPT=$(mktemp)
  trap 'rm -f "$NVM_SCRIPT"' EXIT
  curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" -o "$NVM_SCRIPT"
  if command -v sha256sum &>/dev/null; then
    echo "${NVM_CHECKSUM}  ${NVM_SCRIPT}" | sha256sum -c -
  elif command -v shasum &>/dev/null; then
    echo "${NVM_CHECKSUM}  ${NVM_SCRIPT}" | shasum -a 256 -c -
  else
    echo "No sha256 tool found; cannot verify nvm checksum"; exit 1
  fi || { echo "nvm checksum verification failed"; exit 1; }
  bash "$NVM_SCRIPT"

  # Source nvm script in the current session
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

# Install + use the pinned Node.js version. `nvm use` alone silently no-ops
# when v20 isn't installed in the current shell (e.g. non-interactive CI),
# and the build then runs against whatever node is on PATH. Install first, use
# second, and verify `node -v` actually reports v20 before continuing.
if ! nvm install v20; then
  echo "Error: nvm install v20 failed. Deployment aborted."
  exit 1
fi

if ! nvm use v20; then
  echo "Error: Failed to switch to Node.js v20. Deployment aborted."
  exit 1
fi

NODE_VER="$(node -v 2>/dev/null || true)"
case "$NODE_VER" in
  v20.*) echo "Confirmed Node.js $NODE_VER" ;;
  *)
    echo "Error: expected Node.js v20.* but got '$NODE_VER'. Deployment aborted."
    exit 1
    ;;
esac

# print contents of ui_colors.json
echo "Contents of ui_colors.json:"
cat ui_colors.json

# Run npm build
npm run build

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Build successful. Copying files..."

  # echo current dir
  echo
  pwd

  # Specify the destination directory
  destination_dir="../../litellm/proxy/_experimental/out"

  # Remove existing files in the destination directory
  rm -rf "$destination_dir"/*

  # Copy the contents of the output directory to the specified destination
  cp -r ./out/* "$destination_dir"

  rm -rf ./out

  echo "Deployment completed."
else
  echo "Build failed. Deployment aborted."
  exit 1
fi

