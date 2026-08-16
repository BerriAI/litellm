# Release Process

This document describes the release process for the LiteLLM Terraform Provider.

## Overview

Releases are automated via GitHub Actions when a version tag is pushed. The workflow builds the provider for multiple platforms, signs the artifacts with GPG, and publishes them to GitHub Releases.

## Prerequisites

### GPG Key Setup (One-Time Setup for Repository Maintainers)

The Terraform Registry requires all providers to be signed with a GPG key. This must be configured before the first release.

#### 1. Generate a GPG Key

If you don't already have a GPG key for provider signing:

```bash
gpg --full-generate-key
```

Configuration:
- Key type: RSA and RSA (default)
- Key size: 4096 bits
- Expiration: No expiration (or set a long expiration period)
- Email: Use an email associated with your GitHub account
- Set a strong passphrase (or leave empty for CI/CD use)

#### 2. Export the GPG Key

```bash
# List your keys to get the key ID
gpg --list-secret-keys --keyid-format=long

# Example output:
# sec   rsa4096/ABCD1234EFGH5678 2024-01-01 [SC]
#       1234567890ABCDEF1234567890ABCDEF12345678
# uid                 [ultimate] Your Name <your.email@example.com>
#
# The key ID is: ABCD1234EFGH5678
# The fingerprint is: 1234567890ABCDEF1234567890ABCDEF12345678

# Export the private key (ASCII-armored format)
gpg --armor --export-secret-keys ABCD1234EFGH5678

# Export the public key
gpg --armor --export ABCD1234EFGH5678
```

#### 3. Configure GitHub Repository Secrets

Add the following secrets to the repository at: **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Description | Value |
|-------------|-------------|-------|
| `GPG_PRIVATE_KEY` | The GPG private key for signing releases | Full output from `gpg --armor --export-secret-keys` (including `-----BEGIN PGP PRIVATE KEY BLOCK-----` and `-----END PGP PRIVATE KEY BLOCK-----`) |
| `PASSPHRASE` | The passphrase for the GPG key | Your GPG key passphrase (leave empty if no passphrase was set) |

#### 4. Register Public Key with Terraform Registry

Before publishing to the Terraform Registry:

1. Go to https://registry.terraform.io/settings/gpg-keys
2. Click "Add a key"
3. Paste your public GPG key (output from `gpg --armor --export`)
4. Submit

**Note**: The public key fingerprint must match the key used to sign the provider releases.

## Release Steps

### 1. Prepare the Release

Before creating a release:

1. **Update CHANGELOG.md**
   - Move items from `[Unreleased]` section to a new version section
   - Follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
   - Use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for version numbers
   - Include all notable changes since the last release

   Example:
   ```markdown
   ## [0.1.2] - 2026-02-20

   ### Added
   - New feature description

   ### Fixed
   - Bug fix description

   ### Changed
   - Changed behavior description
   ```

2. **Verify tests pass**
   ```bash
   make test
   ```

3. **Verify the build works locally**
   ```bash
   make build
   ```

4. **Land the changes in BerriAI/litellm**

   Open a PR to `BerriAI/litellm` updating `terraform/provider/CHANGELOG.md` (and any source changes) and merge it. Note the merge commit SHA; the release workflow takes it as `git_ref`

### 2. Mirror and Tag via project-releaser

The provider source lives at `terraform/provider/` in `BerriAI/litellm`; `BerriAI/terraform-provider-litellm` is a thin release mirror. Do not commit or tag the mirror directly

1. Go to `BerriAI/project-releaser` > **Actions** > `Publish Terraform provider`
2. Click **Run workflow**:
   - `git_ref`: full 40-char commit SHA from `BerriAI/litellm` to release from
   - `provider_version`: the new version without the `v` prefix (e.g. `0.3.0`)
   - `dry_run`: optional; validates without pushing
3. The workflow rsyncs `terraform/provider/` into the mirror repo, commits, and pushes tag `v<provider_version>`
4. The tag push triggers the mirror's `Release` workflow (goreleaser), which is gated by the `production-release` environment approval

**Important**:
- Tags must follow the format: `v<MAJOR>.<MINOR>.<PATCH>` (e.g., `v0.1.2`, `v1.0.0`)
- The workflow refuses to overwrite an existing tag; publish a new version instead

### 3. Monitor the Release Workflow

1. Go to: https://github.com/BerriAI/terraform-provider-litellm/actions
2. Find the "Release" workflow run for your tag
3. Monitor the progress and check for any errors

The workflow will:
- Check out the code
- Set up Go
- Import the GPG key
- Run `go mod tidy`
- Build binaries for multiple platforms (Linux, macOS, Windows, FreeBSD)
- Create archives and checksums
- Sign the checksums with GPG
- Create a GitHub release
- Upload all artifacts

### 4. Verify the Release

After the workflow completes successfully:

1. **Check the GitHub Release**
   - Go to: https://github.com/BerriAI/terraform-provider-litellm/releases
   - Verify the release was created with the correct version
   - Confirm all artifacts are present:
     - Binary archives for each platform
     - SHA256SUMS file
     - SHA256SUMS.sig (GPG signature)
     - terraform-registry-manifest.json

2. **Verify the signature** (optional)
   ```bash
   # Download the checksums and signature
   wget https://github.com/BerriAI/terraform-provider-litellm/releases/download/v0.1.2/terraform-provider-litellm_0.1.2_SHA256SUMS
   wget https://github.com/BerriAI/terraform-provider-litellm/releases/download/v0.1.2/terraform-provider-litellm_0.1.2_SHA256SUMS.sig

   # Verify the signature
   gpg --verify terraform-provider-litellm_0.1.2_SHA256SUMS.sig terraform-provider-litellm_0.1.2_SHA256SUMS
   ```

### 5. Publish to Terraform Registry (Optional)

If this provider is published to the Terraform Registry:

1. The registry should automatically detect the new release via the GitHub webhook
2. If not, you may need to manually trigger a sync on the Terraform Registry dashboard
3. Verify the new version appears at: https://registry.terraform.io/providers/BerriAI/litellm/latest

## Troubleshooting

### Release Workflow Fails with GPG Error

**Error**: `Input required and not supplied: gpg_private_key`

**Solution**:
- Verify that `GPG_PRIVATE_KEY` and `PASSPHRASE` secrets are configured in the repository
- Ensure the secrets are not expired
- Check that the secret names match exactly (case-sensitive)

### GoReleaser Signing Fails

**Error**: `gpg: signing failed: No secret key`

**Solution**:
- Verify the `GPG_PRIVATE_KEY` secret contains the complete private key block
- Ensure the passphrase is correct
- Check that the key hasn't expired: `gpg --list-keys`

### Build Fails

**Error**: Build errors during compilation

**Solution**:
- Run `make test` and `make build` locally first
- Ensure `go.mod` and `go.sum` are up to date
- Check that all dependencies are available

### Tag Already Exists

**Error**: The publish workflow refuses to push because the tag already exists on the mirror

**Solution**: Tags are immutable by design. Re-run the workflow with a new patch version instead of deleting or moving an existing tag

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): New functionality in a backward-compatible manner
- **PATCH** version (0.0.1): Backward-compatible bug fixes

For pre-1.0 releases:
- Breaking changes may occur in minor versions
- Patch versions should only contain bug fixes

## Security Considerations

1. **Never commit private keys**: The GPG private key should only be stored as a GitHub secret
2. **Protect repository secrets**: Limit who has access to manage repository secrets
3. **Use a dedicated key**: Consider using a separate GPG key specifically for provider signing
4. **Key rotation**: If the GPG key is compromised, generate a new key, update secrets, and register the new public key with the Terraform Registry
5. **Passphrase**: Use a strong passphrase for the GPG key, or use a passphrase-less key specifically for CI/CD

## References

- [GoReleaser Documentation](https://goreleaser.com/)
- [Terraform Provider Publishing](https://www.terraform.io/docs/registry/providers/publishing.html)
- [HashiCorp GPG Signing Requirements](https://www.terraform.io/docs/registry/providers/publishing.html#signing-releases)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
