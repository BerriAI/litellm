# Custom Semgrep Rules

All `.yml` files under `.semgrep/rules/` run in CI (CircleCI `semgrep` job).

## Add a Rule

* Add a `.yml` file under `.semgrep/rules/<language>/<domain>/`


[Rule syntax →](https://semgrep.dev/docs/writing-rules/rule-syntax/)

## Organizing Rules

### Structure: language → domain

```
.semgrep/rules/<language>/<domain>/<rule-name>.yml
```

Examples:

- `python/security/unsafe-yaml-load.yml`
- `python/reliability/missing-timeout-http.yml`
- `python/performance/blocking-io-in-async.yml`

### Rule metadata

Match tags to the folder for consistent filtering:

```yaml
metadata:
  tags: [python, security]
```

### Severity expectations

All rules must fail CI on findings. No warn-only rules.

- Use `severity: ERROR` in rule metadata
- If a rule is noisy → refine until low false positives before adding

## Run Locally

```bash
semgrep scan --config .semgrep/rules . --error
```

With Semgrep registry:

```bash
semgrep scan --config auto --config .semgrep/rules .
```
