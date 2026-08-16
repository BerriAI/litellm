#!/bin/bash

# Check if UI_BASE_PATH argument is provided
if [ -z "$1" ]; then
    echo "Error: UI_BASE_PATH argument is required."
    echo "Usage: $0 <UI_BASE_PATH>"
    exit 1
fi

# Set UI_BASE_PATH from the first argument
UI_BASE_PATH="$1"

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

# Use nvm to set the required Node.js version
nvm use v18.17.0

# Check if nvm use was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to switch to Node.js v18.17.0. Deployment aborted."
    exit 1
fi

# Run npm build with the environment variable
UI_BASE_PATH=$UI_BASE_PATH npm run build

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

    echo "Deployment completed."
else
    echo "Build failed. Deployment aborted."
fi