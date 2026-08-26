# MCP docs golden-path snippets

These files mirror the golden-path examples in the LiteLLM docs (BerriAI/litellm-docs). `tests/mcp_tests/test_mcp_docs_golden_path.py` parses each snippet and executes it against a live in-process proxy backed by a local mock MCP server (`places_mcp_server.py`), so CI on this repo fails when an endpoint, required header, tool-name convention, or config field the docs promise stops working

Each snippet's first `# source:` line names the doc page and section it mirrors. Test failures print that line, so a red run points at the exact doc example that drifted. Example values (`places_api`, `getPlaces`, `sk-1234`) match the example server the docs use

Run locally before opening a docs or MCP PR:

```bash
make test-mcp-docs
```

Provider-backed snippets (`responses_embedded.sh`, `chat_completions_mcp.sh`) call a real model and are skipped explicitly unless `OPENAI_API_KEY` is set. Everything else runs offline with no secrets and no third-party MCP service

Ownership split with the docs repo: this suite (BerriAI/litellm, owner: proxy/MCP maintainers) validates that documented gateway behavior still works, and runs in `.github/workflows/test-mcp.yml`. BerriAI/litellm-docs owns prose, internal links, and anchors: its `npm run build` (Docusaurus) validates MDX, links, and anchors in its own CI. If this suite goes red because the docs changed their golden-path example, update the snippet here in the same PR that changes the behavior, or ask the docs change to be reverted; the committer of the breaking change owns the fix
