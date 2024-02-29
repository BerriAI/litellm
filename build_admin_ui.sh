#!/bin/bash

# # try except this script
# set -e

# print current dir 
echo
pwd


# only run this step for litellm enterprise, we run this if enterprise/enterprise_ui/_enterprise.json exists
# Check if the path to enterprise_colors.json is provided as an env variable
if [ -z "$ENTERPRISE_COLORS_PATH" ]; then
    # If not provided, use the default path
    ENTERPRISE_COLORS_PATH="enterprise/enterprise_ui/enterprise_colors.json"
fi

# Check if enterprise_colors.json exists at the specified path
if [ ! -f "$ENTERPRISE_COLORS_PATH" ]; then
    echo "Admin UI - using default LiteLLM UI"
    exit 0
fi

echo "Building Custom Admin UI..."

# Install dependencies
# Check if we are on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    # Install dependencies using Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew not found. Please install Homebrew and try again."
        exit 1
    fi
    brew update
    brew install curl
else
    # Assume Linux, try using apt-get
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y curl
    elif command -v apk &> /dev/null; then
    # Try using apk if apt-get is not available
        apk update
        apk add curl
    else
        echo "Error: Unsupported package manager. Cannot install dependencies."
        exit 1
    fi
fi
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
source ~/.nvm/nvm.sh
nvm install v18.17.0
nvm use v18.17.0
npm install -g npm

# Copy enterprise_colors.json from the specified path to /ui/litellm-dashboard, and rename it to ui_colors.json
cp "$ENTERPRISE_COLORS_PATH" ui/litellm-dashboard/ui_colors.json

# Change directory to /ui/litellm-dashboard
cd ui/litellm-dashboard

# Ensure access to build_ui.sh
chmod +x ./build_ui.sh

# Run ./build_ui.sh
./build_ui.sh

# Return to the root directory
cd ../..