# Security Scans

## Scans that run:

- Trivy scan on `./docs/` (HIGH/CRITICAL/MEDIUM)
- Trivy scan on `./ui/` (HIGH/CRITICAL/MEDIUM) 
- Grype scan on `Dockerfile.database` (fails on CRITICAL)
- Grype scan on main `Dockerfile` (fails on CRITICAL)
- Grype CVSS ≥ 4.0 scan on main `Dockerfile` (fails any vulnerabilities with CVSS ≥ 4.0)
