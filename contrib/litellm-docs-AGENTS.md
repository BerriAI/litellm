# Copy to BerriAI/litellm-docs root as `AGENTS.md`

Open a PR in https://github.com/BerriAI/litellm-docs with this file at the repository root.

---

# INSTRUCTIONS FOR LITELLM-DOCS

User-facing documentation for [LiteLLM](https://github.com/BerriAI/litellm). Published at https://docs.litellm.ai.

## Companion code repo

- Implementation and behavior live in **BerriAI/litellm** (proxy, SDK, providers).
- When documenting features, verify behavior against the `litellm` code repo.
- When answering questions in a multi-root workspace or Cloud multi-repo environment, search **both** this repo and `litellm`.

## Cursor setup

- **Desktop:** open `litellm-full.code-workspace` from the `litellm` repo (adds this folder as `litellm-docs`).
- **Cloud:** use a multi-repo environment with `BerriAI/litellm` and `BerriAI/litellm-docs`.
- **Published site:** add https://docs.litellm.ai under Cursor Settings → Docs for `@Docs` context.

## Contributing

- Doc-only changes belong in this repository.
- Do not duplicate long implementation detail here; link to https://docs.litellm.ai and keep code references accurate against `litellm`.
