# Upstream workflow backup (inactive)

These files were copied from `.github/workflows/` when trimming the Bitovi fork.

GitHub Actions **only** runs workflows in `.github/workflows/`. Files here are kept for reference and easy restore; they do not run.

## Active workflow

Only this file remains enabled:

- `../workflows/publish-bitovi-proxy-image.yml` — build and push the proxy image to ECR

Deploy, rollback, and ECR cutover: [DEPLOYMENT.md](../../DEPLOYMENT.md) in this repo. Infra and config live in `bitovi/claude-usage-proxy`.

## Restore a workflow

```bash
mv .github/workflows-backup/<workflow>.yml .github/workflows/
```

## Restore everything (not recommended for the fork)

```bash
mv .github/workflows-backup/*.{yml,yaml,py} .github/workflows/
```

## Other disabled automation

- `.github/dependabot.yaml.bak` — renamed from `dependabot.yaml` so Dependabot does not open daily action-bump PRs. Restore with `mv .github/dependabot.yaml.bak .github/dependabot.yaml` if needed.
