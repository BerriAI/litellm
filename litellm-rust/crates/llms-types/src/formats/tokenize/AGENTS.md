# rules

- The unified endpoint accepts text in `prompt` and keeps numeric `token_ids` separate from string `tokens`
- Missing provider token representations serialize as null, never fabricated IDs or empty arrays

# references

- Provider endpoint contracts live in `../../../../llms/src/hosted_vllm/tokenize/AGENTS.md` and `../../../../llms/src/watsonx/tokenize/AGENTS.md`
