#!/bin/bash

# Script to create a branch with litellm_ prefix from a contributor's branch
# Usage: ./create_litellm_branch.sh [source_branch] [new_branch_name]
# 
# Examples:
#   ./create_litellm_branch.sh branch-name
#   ./create_litellm_branch.sh remote:branch-name
#   ./create_litellm_branch.sh codgician:ghcopilot-costmap
#
# If source_branch is in format "remote:branch", the script will:
#   - Automatically add the remote if it doesn't exist (assumes GitHub fork)
#   - Fetch the branch from that remote
#   - Create a new branch with litellm_ prefix
#
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

# Handle remote:branch format (e.g., codgician:ghcopilot-costmap)
if [[ "$SOURCE_BRANCH" == *:* ]]; then
    REMOTE_NAME="${SOURCE_BRANCH%%:*}"
    BRANCH_NAME="${SOURCE_BRANCH#*:}"
    
    print_info "Detected remote:branch format - remote: '$REMOTE_NAME', branch: '$BRANCH_NAME'"
    
    # Check if remote exists
    if ! git remote | grep -q "^${REMOTE_NAME}$"; then
        print_info "Remote '$REMOTE_NAME' not found. Attempting to add it..."
        
        # Try to add remote (assuming GitHub fork)
        if git remote add "$REMOTE_NAME" "https://github.com/${REMOTE_NAME}/litellm.git" 2>/dev/null; then
            print_success "Added remote '$REMOTE_NAME'"
        else
            print_error "Failed to add remote '$REMOTE_NAME'. Please add it manually:"
            print_info "  git remote add $REMOTE_NAME https://github.com/${REMOTE_NAME}/litellm.git"
            exit 1
        fi
    fi
    
    # Fetch the branch from the remote
    print_info "Fetching branch '$BRANCH_NAME' from remote '$REMOTE_NAME'..."
    if git fetch "$REMOTE_NAME" "$BRANCH_NAME":"$BRANCH_NAME" 2>/dev/null; then
        print_success "Fetched branch '$BRANCH_NAME' from '$REMOTE_NAME'"
        SOURCE_BRANCH="$BRANCH_NAME"
    else
        # Try fetching without creating local branch
        if git fetch "$REMOTE_NAME" "$BRANCH_NAME" 2>/dev/null; then
            print_success "Fetched branch '$BRANCH_NAME' from '$REMOTE_NAME'"
            SOURCE_BRANCH="$REMOTE_NAME/$BRANCH_NAME"
        else
            print_error "Failed to fetch branch '$BRANCH_NAME' from remote '$REMOTE_NAME'"
            print_info "Please verify the remote and branch name exist"
            exit 1
        fi
    fi
fi

# Get new branch name
if [ -n "$2" ]; then
    NEW_BRANCH_NAME="$2"
else
    # Auto-generate from source branch name (use just branch name, not remote/branch)
    if [[ "$SOURCE_BRANCH" == */* ]]; then
        NEW_BRANCH_NAME="${SOURCE_BRANCH##*/}"
    else
        NEW_BRANCH_NAME="$SOURCE_BRANCH"
    fi
fi

# Remove litellm_ prefix if it already exists
if [[ "$NEW_BRANCH_NAME" == litellm_* ]]; then
    NEW_BRANCH_NAME="${NEW_BRANCH_NAME#litellm_}"
    print_warning "Removed existing litellm_ prefix from branch name"
fi

# Add litellm_ prefix
NEW_BRANCH_NAME="litellm_${NEW_BRANCH_NAME}"


# Function to check if branch exists in any remote
branch_exists_in_remote() {
    local branch_name="$1"
    # Check local branches
    if git show-ref --verify --quiet refs/heads/"$branch_name"; then
        return 0
    fi
    # Check all remote branches
    for remote in $(git remote); do
        if git show-ref --verify --quiet refs/remotes/"$remote"/"$branch_name"; then
            return 0
        fi
    done
    return 1
}

# Check if source branch exists
if ! branch_exists_in_remote "$SOURCE_BRANCH"; then
    print_error "Source branch '$SOURCE_BRANCH' does not exist locally or in any remote"
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
        # Find which remote has the branch
        REMOTE_WITH_BRANCH=""
        for remote in $(git remote); do
            if git show-ref --verify --quiet refs/remotes/"$remote"/"$SOURCE_BRANCH"; then
                REMOTE_WITH_BRANCH="$remote"
                break
            fi
        done
        
        if [ -n "$REMOTE_WITH_BRANCH" ]; then
            print_info "Source branch '$SOURCE_BRANCH' exists in remote '$REMOTE_WITH_BRANCH'"
            # Use the remote branch directly
            SOURCE_BRANCH="$REMOTE_WITH_BRANCH/$SOURCE_BRANCH"
        else
            # Try fetching from origin as fallback
            print_info "Fetching source branch '$SOURCE_BRANCH' from remote..."
            git fetch origin "$SOURCE_BRANCH":"$SOURCE_BRANCH" || {
                print_error "Failed to fetch branch '$SOURCE_BRANCH' from remote"
                exit 1
            }
        fi
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

