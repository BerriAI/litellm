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
    # Install nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

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