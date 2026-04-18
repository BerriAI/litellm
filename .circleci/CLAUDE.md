# CI (`.circleci/`)

CircleCI pipeline configuration.

## CI Supply-Chain Safety
- **Never pipe a remote script into a shell** (`curl ... | bash`, `wget ... | sh`). Download the artifact to a file, verify its SHA-256 checksum, then install.
- **Pin every external tool to a specific version** with a full URL (not `latest` or `stable`). Unversioned downloads silently change under you.
- **Verify checksums for all downloaded binaries.** Use the provider's official `.sha256` / `.sha256sum` sidecar file when available; otherwise compute and hardcode the digest.
- **Prefer reusable CircleCI commands** (`commands:` section) so a tool is installed and verified in exactly one place, then referenced everywhere with `- install_<tool>` or `- wait_for_service`.
- **Don't add tools just because they were there before.** Audit whether an external dependency is still needed. If it can be replaced with a shell one-liner or a tool already in the image, remove it.
- These rules apply to every download in CI: binaries, install scripts, language version managers, package repos. No exceptions.
