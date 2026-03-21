# Custom Semgrep rules for LiteLLM

Add custom rule YAML files here. Semgrep loads all `.yml`/`.yaml` files under this directory.

**Run only custom rules (CI / fail on findings):**

```bash
semgrep scan --config .semgrep/rules . --error
```

**Run with registry + custom rules:**

```bash
semgrep scan --config auto --config .semgrep/rules .
```

**Layout:**

- `python/` – Python-specific rules (security, patterns)
- Add more subdirs as needed (e.g. `generic/` for language-agnostic rules)

See [Semgrep rule syntax](https://semgrep.dev/docs/writing-rules/rule-syntax/).
