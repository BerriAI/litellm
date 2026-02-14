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

# Function to install ggshield
install_ggshield() {
    echo "Installing ggshield..."
    pip3 install --upgrade pip
    pip3 install ggshield
    echo "ggshield installed successfully"
}

# # Function to run secret detection scans
# run_secret_detection() {
#     echo "Running secret detection scans..."
    
#     if ! command -v ggshield &> /dev/null; then
#         install_ggshield
#     fi
    
#     # Check if GITGUARDIAN_API_KEY is set (required for CI/CD)
#     if [ -z "$GITGUARDIAN_API_KEY" ]; then
#         echo "Warning: GITGUARDIAN_API_KEY environment variable is not set."
#         echo "ggshield requires a GitGuardian API key to scan for secrets."
#         echo "Please set GITGUARDIAN_API_KEY in your CI/CD environment variables."
#         exit 1
#     fi
    
#     echo "Scanning codebase for secrets..."
#     echo "Note: Large codebases may take several minutes due to API rate limits (50 requests/minute on free plan)"
#     echo "ggshield will automatically handle rate limits and retry as needed."
#     echo "Binary files, cache files, and build artifacts are excluded via .gitguardian.yaml"
    
#     # Use --recursive for directory scanning and auto-confirm if prompted
#     # .gitguardian.yaml will automatically exclude binary files, wheel files, etc.
#     # GITGUARDIAN_API_KEY environment variable will be used for authentication
#     echo y | ggshield secret scan path . --recursive || {
#         echo ""
#         echo "=========================================="
#         echo "ERROR: Secret Detection Failed"
#         echo "=========================================="
#         echo "ggshield has detected secrets in the codebase."
#         echo "Please review discovered secrets above, revoke any actively used secrets"
#         echo "from underlying systems and make changes to inject secrets dynamically at runtime."
#         echo ""
#         echo "For more information, see: https://docs.gitguardian.com/secrets-detection/"
#         echo "=========================================="
#         echo ""
#         exit 1
#     }
    
#     echo "Secret detection scans completed successfully"
# }

# Function to run Trivy scans
run_trivy_scans() {
    echo "Running Trivy scans..."
    
    echo "Scanning LiteLLM Docs..."
    trivy fs --ignorefile .trivyignore --scanners vuln --dependency-tree --exit-code 1 --severity HIGH,CRITICAL,MEDIUM ./docs/
    
    echo "Scanning LiteLLM UI..."
    trivy fs --ignorefile .trivyignore --scanners vuln --dependency-tree --exit-code 1 --severity HIGH,CRITICAL,MEDIUM ./ui/
    
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
    grype litellm-database:latest --config ci_cd/.grype.yaml --fail-on critical
    
    # Build and scan main Dockerfile
    echo "Building and scanning main Dockerfile..."
    docker build --no-cache -t litellm:latest .
    grype litellm:latest --config ci_cd/.grype.yaml --fail-on critical
    
    # Restore original .dockerignore
    echo "Restoring original .dockerignore..."
    mv .dockerignore.backup .dockerignore
    
    # Scan the locally built LiteLLM image for vulnerabilities with CVSS >= 4.0
    echo "Scanning locally built LiteLLM image for high-severity vulnerabilities..."
    echo "Using locally built image: litellm:latest"
    
    # Allowlist of CVEs to be ignored in failure threshold/reporting
    # - CVE-2025-8869: Not applicable on Python >=3.13 (PEP 706 implemented); pip fallback unused; no OS-level fix
    # - GHSA-4xh5-x5gv-qwph: GitHub Security Advisory alias for CVE-2025-8869
    # - GHSA-5j98-mcp5-4vw2: glob CLI command injection via -c/--cmd; glob CLI is not used in the litellm runtime image,
    #   and the vulnerable versions are pulled in only via OS-level/node tooling outside of our application code
    ALLOWED_CVES=(
        "CVE-2025-8869"
        "GHSA-4xh5-x5gv-qwph"
        "CVE-2025-8291" # no fix available as of Oct 11, 2025
        "GHSA-5j98-mcp5-4vw2"
        "CVE-2025-13836" # Python 3.13 HTTP response reading OOM/DoS - no fix available in base image
        "CVE-2025-12084" # Python 3.13 xml.dom.minidom quadratic algorithm - no fix available in base image
        "CVE-2025-60876" # BusyBox wget HTTP request splitting - no fix available in Chainguard Wolfi base image
        "CVE-2026-0861" # Wolfi glibc still flagged even on 2.42-r5; upstream patched build unavailable yet
        "CVE-2010-4756" # glibc glob DoS - awaiting patched Wolfi glibc build
        "CVE-2019-1010022" # glibc stack guard bypass - awaiting patched Wolfi glibc build
        "CVE-2019-1010023" # glibc ldd remap issue - awaiting patched Wolfi glibc build
        "CVE-2019-1010024" # glibc ASLR mitigation bypass - awaiting patched Wolfi glibc build
        "CVE-2019-1010025" # glibc pthread heap address leak - awaiting patched Wolfi glibc build
        "CVE-2026-22184" # zlib untgz buffer overflow - untgz unused + no fixed Wolfi build yet
        "GHSA-58pv-8j8x-9vj2" # jaraco.context path traversal - setuptools vendored only (v5.3.0), not used in application code (using v6.1.0+)
        "GHSA-34x7-hfp2-rc4v" # node-tar hardlink path traversal - not applicable, tar CLI not exposed in application code
        "GHSA-r6q2-hw4h-h46w" # node-tar not used by application runtime, Linux-only container, not affect by macOS APFS-specific exploit
        "GHSA-8rrh-rw8j-w5fx" # wheel is from chainguard and will be handled by then TODO: Remove this after Chainguard updates the wheel
        "CVE-2025-59465" # Node only used for Admin UI build/prisma
        "CVE-2025-55131" # Node only used for Admin UI build/prisma
        "CVE-2025-59466" # Node only used for Admin UI build/prisma
        "CVE-2025-55130" # Node only used for Admin UI build/prisma
        "CVE-2025-59467" # Node only used for Admin UI build/prisma
        "CVE-2026-21637" # Node only used for Admin UI build/prisma
        "CVE-2025-55132" # Node only used for Admin UI build/prisma
        "GHSA-hx9q-6w63-j58v" # orjson dumps recursion; allowlisted
        "CVE-2025-15281" # No fix available yet
        "CVE-2026-0865" # No fix available yet
        "CVE-2025-15282" # No fix available yet
        "CVE-2026-0672" # No fix available yet
        "CVE-2025-15366" # No fix available yet
        "CVE-2025-15367" # No fix available yet
        "CVE-2025-12781" # No fix available yet
        "CVE-2025-11468" # No fix available yet
        "CVE-2026-1299" # Python 3.13 email module header injection - not applicable, LiteLLM doesn't use BytesGenerator for email serialization
        "CVE-2026-0775" # npm cli incorrect permission assignment - no fix available yet, npm is only used at build/prisma-generate time
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
    
    # echo "Running secret detection scans..."
    # run_secret_detection
    
    echo "Running filesystem vulnerability scans..."
    run_trivy_scans
    
    echo "Running Docker image vulnerability scans..."
    run_grype_scans
    
    echo "All security scans completed successfully!"
}

# Execute main function
main "$@"
