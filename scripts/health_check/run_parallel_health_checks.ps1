# Parallel LiteLLM Health Check Runner (PowerShell version)
#
# This script runs multiple health check containers in parallel.
#
# Usage:
#   $env:LITELLM_BASE_URL="https://litellm.example.com"
#   $env:LITELLM_API_KEY="your-api-key"
#   .\run_parallel_health_checks.ps1 [num_parallel_jobs] [image_name]
#
# Defaults:
#   - num_parallel_jobs: 16
#   - image_name: litellm/litellm-health-check:latest

param(
    [int]$NumParallelJobs = 16,
    [string]$ImageName = "litellm/litellm-health-check:latest",
    [string]$ContainerRuntime = "docker"
)

# Set defaults for environment variables if not provided
if (-not $env:LITELLM_BASE_URL) {
    $env:LITELLM_BASE_URL = "https://litellm-perf-cache-and-router.onrender.com"
    Write-Warning "LITELLM_BASE_URL not set, using default: $env:LITELLM_BASE_URL"
}

if (-not $env:LITELLM_API_KEY) {
    $env:LITELLM_API_KEY = "sk-1234"
    Write-Warning "LITELLM_API_KEY not set, using default: $env:LITELLM_API_KEY"
}

# Check if container runtime is available
$runtimeExists = Get-Command $ContainerRuntime -ErrorAction SilentlyContinue
if (-not $runtimeExists) {
    Write-Error "Error: $ContainerRuntime is not installed"
    exit 1
}

Write-Host "Running $NumParallelJobs parallel health check containers..." -ForegroundColor Yellow
Write-Host "Using image: $ImageName" -ForegroundColor Yellow
Write-Host "Container runtime: $ContainerRuntime" -ForegroundColor Yellow
Write-Host "LiteLLM Base URL: $env:LITELLM_BASE_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: This will run continuously. Press Ctrl+C to stop." -ForegroundColor Red
Write-Host ""
Write-Host "Troubleshooting:" -ForegroundColor Yellow
Write-Host "  - If you see 'All connection attempts failed', check:" -ForegroundColor Yellow
Write-Host "    1. Is the LiteLLM proxy running on the expected port?" -ForegroundColor Yellow
Write-Host "    2. Set LITELLM_BASE_URL to the correct URL (e.g., http://host.docker.internal:PORT)" -ForegroundColor Yellow
Write-Host "    3. On Linux, you may need to use the host IP instead of host.docker.internal" -ForegroundColor Yellow
Write-Host ""

# Capture environment variables in parent scope for use in parallel block
$baseUrl = $env:LITELLM_BASE_URL
$apiKey = $env:LITELLM_API_KEY
$customAuthHeader = $env:LITELLM_CUSTOM_AUTH_HEADER

# Run parallel health checks
# This creates an infinite loop that keeps spawning containers
# Each container tests all models, then exits, and a new one starts
while ($true) {
    # Start up to NumParallelJobs containers in parallel
    1..$NumParallelJobs | ForEach-Object -Parallel {
        $runtime = $using:ContainerRuntime
        $imageName = $using:ImageName
        $baseUrl = $using:baseUrl
        $apiKey = $using:apiKey
        $customAuthHeader = $using:customAuthHeader
        
        $envVars = @(
            "-e", "LITELLM_BASE_URL=$baseUrl",
            "-e", "LITELLM_API_KEY=$apiKey",
            "-e", "LITELLM_JSON_OUTPUT=true"
        )
        
        if ($customAuthHeader) {
            $envVars += "-e", "LITELLM_CUSTOM_AUTH_HEADER=$customAuthHeader"
        }
        
        & $runtime run --rm $envVars $imageName
    } -ThrottleLimit $NumParallelJobs
}
