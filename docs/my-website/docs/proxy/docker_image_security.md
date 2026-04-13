# Docker Image Security Guide

LiteLLM signs every Docker image published to GHCR with [cosign](https://docs.sigstore.dev/cosign/overview/) starting from **v1.83.0**. This page covers how to verify signatures, enforce verification in CI/CD, and follow recommended deployment patterns.

## Signed images

All image variants published to `ghcr.io/berriai/` are signed with the same cosign key:

| Image | Description |
|---|---|
| `ghcr.io/berriai/litellm` | Core proxy |
| `ghcr.io/berriai/litellm-database` | Proxy with Postgres dependencies |
| `ghcr.io/berriai/litellm-non_root` | Non-root variant |
| `ghcr.io/berriai/litellm-spend_logs` | Spend-logs sidecar |

The signing key was introduced in [commit `0112e53`](https://github.com/BerriAI/litellm/commit/0112e53046018d726492c814b3644b7d376029d0) and the public key is checked into the repository at [`cosign.pub`](https://github.com/BerriAI/litellm/blob/main/cosign.pub).

:::info Enterprise images
Enterprise images (`litellm-ee`) follow the same signing process. Contact [support@berri.ai](mailto:support@berri.ai) to confirm coverage for your specific enterprise image tag.
:::

## Verify image signatures

Install cosign following the [official instructions](https://docs.sigstore.dev/cosign/system_config/installation/).

### Verify with the pinned commit hash (recommended)

A commit hash is cryptographically immutable, making this the strongest verification method:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm:v1.83.0-stable
```

Replace the image reference with any signed variant:

```bash
# litellm-database
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm-database:v1.83.0-stable

# litellm-non_root
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm-non_root:v1.83.0-stable
```

### Verify with a release tag (convenience)

Tags are protected in this repository and resolve to the same key:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/v1.83.0-stable/cosign.pub \
  ghcr.io/berriai/litellm-database:v1.83.0-stable
```

### Expected output

```
The following checks were performed on each of these signatures:
  - The cosign claims were validated
  - The signatures were verified against the specified public key
```

## Enforce verification in CI/CD

### Kubernetes — Sigstore Policy Controller

The [Sigstore Policy Controller](https://docs.sigstore.dev/policy-controller/overview/) rejects pods whose images fail cosign verification.

1. Install the controller:

```bash
helm repo add sigstore https://sigstore.github.io/helm-charts
helm install policy-controller sigstore/policy-controller \
  -n cosign-system --create-namespace
```

2. Create a `ClusterImagePolicy` with the LiteLLM public key:

```yaml
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata:
  name: litellm-signed-images
spec:
  images:
    - glob: "ghcr.io/berriai/litellm*"
  authorities:
    - key:
        data: |
          -----BEGIN PUBLIC KEY-----
          MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEKi4ivqGpE231OGH50PKbqy1Y1Kkb
          POJC8+i2Wko82gBOUCe3M0Vw86H/4rhUhfoYEti4gdJ9wZbYmK0I2EE96g==
          -----END PUBLIC KEY-----
```

3. Label the namespace to enable enforcement:

```bash
kubectl label namespace litellm policy.sigstore.dev/include=true
```

Any pod in that namespace using an unsigned `ghcr.io/berriai/litellm*` image will be rejected at admission.

### GCP — Binary Authorization

[Binary Authorization](https://cloud.google.com/binary-authorization/docs) can enforce cosign signatures on Cloud Run and GKE.

1. Create a cosign-based attestor using the LiteLLM public key:

```bash
# Import the public key into a Cloud KMS keyring or use a PGP/PKIX attestor.
# See: https://cloud.google.com/binary-authorization/docs/creating-attestors-console
```

2. Configure a Binary Authorization policy that requires the attestor for `ghcr.io/berriai/litellm*` images.

3. Enable the policy on your Cloud Run service or GKE cluster.

Refer to the [GCP Binary Authorization docs](https://cloud.google.com/binary-authorization/docs/setting-up) for full setup steps.

### AWS — ECS / ECR

AWS does not natively verify cosign signatures at deploy time. Common approaches:

- **CI/CD gate**: Run `cosign verify` in your deployment pipeline before pushing to ECR or updating the ECS task definition. Fail the pipeline if verification fails.
- **OPA/Gatekeeper on EKS**: If running on EKS, use the Sigstore Policy Controller (same as the Kubernetes approach above).

### GitHub Actions gate

Add a verification step before any deployment job:

```yaml
- name: Verify LiteLLM image signature
  run: |
    cosign verify \
      --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
      ghcr.io/berriai/litellm-database:${{ env.LITELLM_VERSION }}
```

## Recommended deployment patterns

### Pin by digest

Digest pinning guarantees the exact image content regardless of tag mutations:

```yaml
image: ghcr.io/berriai/litellm-database@sha256:<digest>
```

Get the digest after pulling:

```bash
docker inspect --format='{{index .RepoDigests 0}}' \
  ghcr.io/berriai/litellm-database:v1.83.0-stable
```

Cosign verification works with digests too:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm-database@sha256:<digest>
```

### Use stable release tags

If digest pinning is too rigid for your workflow, use `-stable` release tags (e.g. `v1.83.0-stable`). These are immutable release tags that will not be overwritten.

Avoid `main-latest` or `main-stable` in production — these rolling tags point to the most recent build and can change between deployments.

### Safe upgrade checklist

1. **Verify the new image** — run `cosign verify` against the new release tag or digest.
2. **Test in staging** — deploy the verified image to a non-production environment.
3. **Update your pinned reference** — change the digest or tag in your deployment manifest.
4. **Deploy to production** — roll out using your standard deployment process.
5. **Monitor `/health`** — confirm the proxy is healthy after the upgrade.

## Further reading

- [CI/CD v2 announcement](https://docs.litellm.ai/blog/ci-cd-v2-improvements) — background on LiteLLM's signing infrastructure
- [Docker deployment guide](./deploy.md) — full Docker, Helm, and Terraform setup
- [cosign documentation](https://docs.sigstore.dev/cosign/overview/) — cosign usage and key management
- [Sigstore Policy Controller](https://docs.sigstore.dev/policy-controller/overview/) — Kubernetes admission control
