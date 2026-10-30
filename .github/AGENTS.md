# Checkout history

Use shallow checkouts by default. `actions/checkout` defaults to one commit, so omitting `fetch-depth` is fine. For comparisons, fetch only the required commits with `--no-tags --depth=1` and preserve the existing baseline semantics. Fetching two shallow endpoints does not make `git merge-base` work across missing ancestry

Use deeper checkout, `fetch-depth: 0`, or `--unshallow` only when the consuming command needs history. For new or changed exceptions, state the consumer and why a targeted fetch cannot preserve its behavior beside that checkout or fetch. Keep bounded retries and timeouts on network operations, and never skip a required check when its baseline is unavailable

Current exceptions are the depth-two parent diff in `test-linting.yml`, Linear's commit-history scan in `linear-release.yml`, and the historical GitGuardian repository scan in `test-linting.yml`. Do not remove release history or reduce secret-scanning coverage just to make checkout shallow

Mateo requested this default on 2026-10-27 after a full-history unit checkout timed out before tests started
