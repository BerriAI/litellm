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
    
    # Allowlist of CVEs to be ignored in failure threshold/reporting
    # - CVE-2025-8869: Not applicable on Python >=3.13 (PEP 706 implemented); pip fallback unused; no OS-level fix
    # - GHSA-4xh5-x5gv-qwph: GitHub Security Advisory alias for CVE-2025-8869
    ALLOWED_CVES=(
        "CVE-2025-8869"
        "GHSA-4xh5-x5gv-qwph"
        "CVE-2025-8291" # no fix available as of Oct 11, 2025
    )

    # Build JSON array of allowlisted CVE IDs for jq
    ALLOWED_IDS_JSON=$(printf '%s\n' "${ALLOWED_CVES[@]}" | jq -R . | jq -s .)

    echo "Checking for vulnerabilities with CVSS score >= 4.0..."
    echo "Allowlisted CVEs (ignored in threshold): ${ALLOWED_CVES[*]}"
    echo ""
    
    # Show all high-severity vulnerabilities for transparency
    TOTAL_HIGH_SEVERITY=$(grype litellm:latest -o json | jq -r '
        .matches[]
        | select(.vulnerability.cvss[]?.metrics.baseScore >= 4.0)
        | .vulnerability.id' | wc -l)
    
    if [ "$TOTAL_HIGH_SEVERITY" -gt 0 ]; then
        echo "Total vulnerabilities found with CVSS >= 4.0: $TOTAL_HIGH_SEVERITY"
        echo ""
        echo "All high-severity vulnerabilities (including allowlisted):"
        grype litellm:latest -o json | jq --argjson allow "$ALLOWED_IDS_JSON" -r '
        ["Package", "Version", "Vulnerability ID", "CVSS Score", "Allowlisted"],
        (.matches[]
          | select(.vulnerability.cvss[]?.metrics.baseScore >= 4.0)
          | [.artifact.name, .artifact.version, .vulnerability.id, .vulnerability.cvss[0].metrics.baseScore, (if (.vulnerability.id as $id | $allow | index($id)) then "YES" else "NO" end)])
        | @tsv' | column -t -s $'\t'
        echo ""
    fi

    HIGH_SEVERITY_COUNT=$(grype litellm:latest -o json | jq --argjson allow "$ALLOWED_IDS_JSON" -r '
        .matches[]
        | select(.vulnerability.cvss[]?.metrics.baseScore >= 4.0)
        | select((.vulnerability.id as $id | $allow | index($id) | not))
        | .vulnerability.id' | wc -l)
    
    if [ "$HIGH_SEVERITY_COUNT" -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "ERROR: Security Scan Failed"
        echo "=========================================="
        echo "Found $HIGH_SEVERITY_COUNT non-allowlisted vulnerabilities with CVSS score >= 4.0 in litellm:latest"
        echo ""
        echo "These vulnerabilities are NOT in the allowlist and must be addressed."
        echo "Current allowlisted CVEs: ${ALLOWED_CVES[*]}"
        echo ""
        echo "Detailed vulnerability report:"
        echo ""
        grype litellm:latest -o json | jq --argjson allow "$ALLOWED_IDS_JSON" -r '
        ["Package", "Version", "Vulnerability ID", "CVSS Score", "Severity", "Fix Version", "Description"],
        (.matches[]
          | select(.vulnerability.cvss[]?.metrics.baseScore >= 4.0)
          | select((.vulnerability.id as $id | $allow | index($id) | not))
          | [.artifact.name, .artifact.version, .vulnerability.id, .vulnerability.cvss[0].metrics.baseScore, .vulnerability.severity, (.vulnerability.fix.versions[0] // "No fix available"), .vulnerability.description])
        | @tsv' | column -t -s $'\t'
        echo ""
        echo "=========================================="
        echo "Action Required:"
        echo "=========================================="
        echo "1. If a fix is available, update the package to the fixed version"
        echo "2. If the vulnerability is not applicable or has no fix:"
        echo "   - Add the CVE/GHSA ID to ALLOWED_CVES array in ci_cd/security_scans.sh"
        echo "   - Add a comment explaining why it's safe to ignore"
        echo ""
        echo "Note: Some vulnerabilities may have multiple IDs (CVE-XXXX and GHSA-XXXX)."
        echo "Add all relevant IDs to the allowlist if they refer to the same issue."
        echo "=========================================="
        echo ""
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
