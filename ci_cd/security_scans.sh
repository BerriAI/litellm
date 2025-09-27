#!/bin/bash

# Security Scans Script for LiteLLM
# This script runs comprehensive security scans including Trivy and Grype

set -e

echo "Starting security scans for LiteLLM..."

# Function to install Trivy and required tools
install_trivy() {
    echo "Installing Trivy and required tools..."
    sudo apt-get update
    sudo apt-get install -y wget apt-transport-https gnupg lsb-release jq curl
    wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
    echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
    sudo apt-get update
    sudo apt-get install trivy
    echo "Trivy and required tools installed successfully"
}

# Function to install Grype
install_grype() {
    echo "Installing Grype..."
    curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sudo sh -s -- -b /usr/local/bin
    echo "Grype installed successfully"
}

# Function to run Trivy scans
run_trivy_scans() {
    echo "Running Trivy scans..."
    
    echo "Scanning LiteLLM Docs..."
    trivy fs --scanners vuln --dependency-tree --exit-code 1 --severity HIGH,CRITICAL,MEDIUM ./docs/
    
    echo "Scanning LiteLLM UI..."
    trivy fs --scanners vuln --dependency-tree --exit-code 1 --severity HIGH,CRITICAL,MEDIUM ./ui/
    
    echo "Trivy scans completed successfully"
}

# Function to build and scan Docker images with Grype
run_grype_scans() {
    echo "Running Grype scans..."
    
    # Temporarily add wheel files to .dockerignore for security scans
    echo "Temporarily modifying .dockerignore to exclude problematic wheel files..."
    cp .dockerignore .dockerignore.backup 2>/dev/null || touch .dockerignore.backup
    echo "/*.whl" >> .dockerignore
    
    # Build and scan Dockerfile.database
    echo "Building and scanning Dockerfile.database..."
    docker build --no-cache -t litellm-database:latest -f ./docker/Dockerfile.database .
    grype litellm-database:latest --fail-on critical
    
    # Build and scan main Dockerfile
    echo "Building and scanning main Dockerfile..."
    docker build --no-cache -t litellm:latest .
    grype litellm:latest --fail-on critical
    
    # Restore original .dockerignore
    echo "Restoring original .dockerignore..."
    mv .dockerignore.backup .dockerignore
    
    # Scan the locally built LiteLLM image for vulnerabilities with CVSS >= 4.0
    echo "Scanning locally built LiteLLM image for high-severity vulnerabilities..."
    echo "Using locally built image: litellm:latest"
    
    # Run grype scan and check for vulnerabilities with CVSS >= 4.0
    echo "Checking for vulnerabilities with CVSS score >= 4.0..."
    HIGH_SEVERITY_COUNT=$(grype litellm:latest -o json | jq -r '.matches[] | select(.vulnerability.cvss[]?.metrics.baseScore >= 4.0) | .vulnerability.id' | wc -l)
    
    if [ "$HIGH_SEVERITY_COUNT" -gt 0 ]; then
        echo "ERROR: Found $HIGH_SEVERITY_COUNT vulnerabilities with CVSS score >= 4.0 in litellm:latest"
        echo "Detailed vulnerability report:"
        grype litellm:latest -o json | jq -r '
        ["Package", "Version", "Vulnerability ID", "CVSS Score", "Severity", "Fix Version", "Description"],
        (.matches[] | select(.vulnerability.cvss[]?.metrics.baseScore >= 4.0) |
        [.artifact.name, .artifact.version, .vulnerability.id, .vulnerability.cvss[0].metrics.baseScore, .vulnerability.severity, (.vulnerability.fix.versions[0] // "No fix available"), .vulnerability.description]) |
        @tsv' | column -t -s $'\t'
        exit 1
    else
        echo "No high-severity vulnerabilities (CVSS >= 4.0) found in litellm:latest"
    fi
    
    echo "Grype scans completed successfully"
}

# Main execution
main() {
    echo "Installing security scanning tools..."
    install_trivy
    install_grype
    
    echo "Running filesystem vulnerability scans..."
    run_trivy_scans
    
    echo "Running Docker image vulnerability scans..."
    run_grype_scans
    
    echo "All security scans completed successfully!"
}

# Execute main function
main "$@"
