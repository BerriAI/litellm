# PowerShell script to create a branch with litellm_ prefix from a contributor's branch
# Usage: .\create_litellm_branch.ps1 [source_branch] [new_branch_name]
# If no arguments provided, uses current branch as source

param(
    [string]$SourceBranch = "",
    [string]$NewBranchName = ""
)

# Function to print colored output
function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Get source branch (default to current branch)
if ([string]::IsNullOrEmpty($SourceBranch)) {
    $SourceBranch = git branch --show-current
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to get current branch"
        exit 1
    }
}

# Get new branch name
if ([string]::IsNullOrEmpty($NewBranchName)) {
    $NewBranchName = $SourceBranch
}

# Remove litellm_ prefix if it already exists
if ($NewBranchName -like "litellm_*") {
    $NewBranchName = $NewBranchName -replace "^litellm_", ""
    Write-Warning "Removed existing litellm_ prefix from branch name"
}

# Add litellm_ prefix
$NewBranchName = "litellm_$NewBranchName"

# Validate branch name (Git branch naming rules)
if ($NewBranchName -notmatch '^[a-zA-Z0-9/._-]+$') {
    Write-Error "Invalid branch name: $NewBranchName"
    Write-Info "Branch names can only contain alphanumeric characters, /, ., _, and -"
    exit 1
}

# Check if source branch exists
$sourceExists = $false
git show-ref --verify --quiet "refs/heads/$SourceBranch" 2>$null
if ($LASTEXITCODE -eq 0) {
    $sourceExists = $true
} else {
    git show-ref --verify --quiet "refs/remotes/origin/$SourceBranch" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $sourceExists = $true
    }
}

if (-not $sourceExists) {
    Write-Error "Source branch '$SourceBranch' does not exist locally or remotely"
    exit 1
}

# Check if new branch already exists
git show-ref --verify --quiet "refs/heads/$NewBranchName" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Warning "Branch '$NewBranchName' already exists locally"
    $response = Read-Host "Do you want to switch to it? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        git checkout $NewBranchName
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Switched to existing branch '$NewBranchName'"
        } else {
            Write-Error "Failed to switch to branch '$NewBranchName'"
            exit 1
        }
        exit 0
    } else {
        Write-Info "Aborted"
        exit 1
    }
}

# Check if we're on the source branch or need to fetch it
$CurrentBranch = git branch --show-current
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to get current branch"
    exit 1
}

if ($CurrentBranch -ne $SourceBranch) {
    # Check if source branch exists locally
    git show-ref --verify --quiet "refs/heads/$SourceBranch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Fetching source branch '$SourceBranch' from remote..."
        git fetch origin "$SourceBranch`:$SourceBranch"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to fetch branch '$SourceBranch' from remote"
            exit 1
        }
    }
}

# Create new branch from source
Write-Info "Creating branch '$NewBranchName' from '$SourceBranch'..."
git checkout -b $NewBranchName $SourceBranch
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create branch '$NewBranchName'"
    exit 1
}

Write-Success "Created and switched to branch '$NewBranchName'"
Write-Info "Source branch: $SourceBranch"
Write-Info "New branch: $NewBranchName"

# Show branch status
Write-Host ""
Write-Info "Branch status:"
git status --short

