
# Harbor

[Harbor](https://github.com/laude-institute/harbor) is a framework from the creators of Terminal-Bench for evaluating and optimizing agents and language models. It uses LiteLLM to call 100+ LLM providers.

```bash
# Install
pip install harbor

# Run a benchmark with any LiteLLM-supported model
harbor run --dataset terminal-bench@2.0 \
   --agent claude-code \
   --model anthropic/claude-opus-4-1 \
   --n-concurrent 4
```

Key features:
- Evaluate agents like Claude Code, OpenHands, Codex CLI
- Build and share benchmarks and environments
- Run experiments in parallel across cloud providers (Daytona, Modal)
- Generate rollouts for RL optimization

- [GitHub](https://github.com/laude-institute/harbor)
- [Documentation](https://harborframework.com/docs)
