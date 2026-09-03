# Release Process

This document describes the release process for the LiteLLM Terraform Provider.

## Overview

The provider is released **in lockstep with LiteLLM**: every LiteLLM release (dev, rc and stable) publishes the provider at the LiteLLM version, built from the same commit as the proxy. There is no separate provider release to cut.

The flow, end to end:

1. `BerriAI/project-releaser`'s release pipeline resolves the commit to release (`main` HEAD for dev; `main` HEAD or an operator-supplied SHA for rc/stable) and passes the release approval gate
2. Its componentized terraform job rsyncs `terraform/provider/` from that commit into `BerriAI/terraform-provider-litellm`, commits, and pushes the tag `v<litellm version>` (for example `v1.99.0`, `v1.99.0-rc.1`, `v1.99.0-dev.1`), alongside the `terraform-aws-litellm` / `terraform-google-litellm` module mirrors which get the same tag
3. The tag push triggers the mirror's own `Release` workflow (goreleaser): multi-platform build, GPG-signed checksums, GitHub release. It runs unattended; project-releaser does not wait for it
4. The public Terraform Registry ingests the GitHub release as provider version `<litellm version>`

`terraform/provider/` only exists from LiteLLM ~1.95, so a stable patch cut from an older line skips the provider and publishes only the modules.

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

## What a change needs

1. **Land it in `BerriAI/litellm`.** Open a PR against `litellm_internal_staging` with the source change and a `CHANGELOG.md` entry under `[Unreleased]`. CI runs `gofmt`, `go vet`, build, tests and the endpoint-drift audit. A change that breaks existing configurations or state must say so in the changelog: the version number cannot signal it any more
2. **Wait for the next LiteLLM release.** The nightly dev release carries it within a day; it reaches a stable version on the next stable cut
3. **Verify** (optional): the version appears at https://registry.terraform.io/providers/BerriAI/litellm and https://github.com/BerriAI/terraform-provider-litellm/releases. If the tag is on the mirror but there is no release, the goreleaser run failed: https://github.com/BerriAI/terraform-provider-litellm/actions

Locally, before opening the PR:

```bash
make test
make build
```

## Out-of-band publish or recovery

Dispatch `Build and Publish Componentized Images + Chart` in `BerriAI/project-releaser` by hand with only `publish_terraform` enabled and the `git_ref` / `tag` of the release to (re)publish. The run waits on project-releaser's release approval, then mirrors and tags exactly as the pipeline does.

The mirror is push-only: do not commit or tag `BerriAI/terraform-provider-litellm` directly. The publish refuses to overwrite an existing tag; a version that failed in goreleaser is recovered by re-running the mirror's `Release` workflow for that tag, not by re-tagging.

The mirror's `.github/` directory (the `Release` workflow) is the one thing the rsync preserves, so a change to the goreleaser *workflow* is a direct PR on the mirror; a change to `.goreleaser.yml` itself lands here like any other source change.

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

**Error**: The publish job refuses to push because the tag already exists on the mirror

**Solution**: Tags are immutable by design and the version is the LiteLLM version, so this means the provider was already mirrored for this release. If the registry is missing the version, re-run the mirror's `Release` workflow for the existing tag rather than re-tagging

## Version Numbering

The provider version is the LiteLLM version, verbatim: `X.Y.Z` for a stable release, `X.Y.Z-rc.N` for a release candidate and `X.Y.Z-dev.N` for a nightly. It says which proxy the provider shipped with and was audited against; it does not follow SemVer's break-signalling, so breaking changes are announced in `CHANGELOG.md` and the registry docs instead.

Versions `0.1.0` to `0.4.0` predate this and remain in the registry on their own line. A `~> 0.4` constraint never receives another release.

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
- [Keep a Changelog](https://keepachangelog.com/)
