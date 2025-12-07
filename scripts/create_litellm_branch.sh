#!/bin/bash

# Script to create a branch with litellm_ prefix from a contributor's branch
# Usage: ./create_litellm_branch.sh [source_branch] [new_branch_name]
# If no arguments provided, uses current branch as source

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Get source branch (default to current branch)
SOURCE_BRANCH="${1:-$(git branch --show-current)}"

# Get new branch name
if [ -n "$2" ]; then
    NEW_BRANCH_NAME="$2"
else
    # Auto-generate from source branch name
    NEW_BRANCH_NAME="$SOURCE_BRANCH"
fi

# Remove litellm_ prefix if it already exists
if [[ "$NEW_BRANCH_NAME" == litellm_* ]]; then
    NEW_BRANCH_NAME="${NEW_BRANCH_NAME#litellm_}"
    print_warning "Removed existing litellm_ prefix from branch name"
fi

# Add litellm_ prefix
NEW_BRANCH_NAME="litellm_${NEW_BRANCH_NAME}"

# Validate branch name (Git branch naming rules)
if ! [[ "$NEW_BRANCH_NAME" =~ ^[a-zA-Z0-9/._-]+$ ]]; then
    print_error "Invalid branch name: $NEW_BRANCH_NAME"
    print_info "Branch names can only contain alphanumeric characters, /, ., _, and -"
    exit 1
fi

# Check if source branch exists
if ! git show-ref --verify --quiet refs/heads/"$SOURCE_BRANCH" && ! git show-ref --verify --quiet refs/remotes/origin/"$SOURCE_BRANCH"; then
    print_error "Source branch '$SOURCE_BRANCH' does not exist locally or remotely"
    exit 1
fi

# Check if new branch already exists
if git show-ref --verify --quiet refs/heads/"$NEW_BRANCH_NAME"; then
    print_warning "Branch '$NEW_BRANCH_NAME' already exists locally"
    read -p "Do you want to switch to it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout "$NEW_BRANCH_NAME"
        print_success "Switched to existing branch '$NEW_BRANCH_NAME'"
        exit 0
    else
        print_info "Aborted"
        exit 1
    fi
fi

# Check if we're on the source branch or need to fetch it
CURRENT_BRANCH=$(git branch --show-current)

if [ "$CURRENT_BRANCH" != "$SOURCE_BRANCH" ]; then
    # Check if source branch exists locally
    if git show-ref --verify --quiet refs/heads/"$SOURCE_BRANCH"; then
        print_info "Source branch '$SOURCE_BRANCH' exists locally"
    else
        print_info "Fetching source branch '$SOURCE_BRANCH' from remote..."
        git fetch origin "$SOURCE_BRANCH":"$SOURCE_BRANCH" || {
            print_error "Failed to fetch branch '$SOURCE_BRANCH' from remote"
            exit 1
        }
    fi
fi

# Create new branch from source
print_info "Creating branch '$NEW_BRANCH_NAME' from '$SOURCE_BRANCH'..."
git checkout -b "$NEW_BRANCH_NAME" "$SOURCE_BRANCH"

print_success "Created and switched to branch '$NEW_BRANCH_NAME'"
print_info "Source branch: $SOURCE_BRANCH"
print_info "New branch: $NEW_BRANCH_NAME"

# Show branch status
echo
print_info "Branch status:"
git status --short

