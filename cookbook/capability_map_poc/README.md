# Capability map POC

Experiment code for learning a model x task capability matrix from tau2-bench telecom traffic. Not used by the proxy

Run every command from `cookbook/` with Python 3.12 and `pydantic`, `openai`, `datasets`, `pytest` installed

```bash
python -m pytest capability_map_poc/tests

# Stage 1: freeze tau2's official telecom train/test split
python -m capability_map_poc.split --split-file "$TAU2_DATA_DIR/tau2/domains/telecom/split_tasks.json" --out-dir out/stage1

# Stage 4a: check the correction labeler against the EMNLP 2025 human labels, through the LiteLLM gateway
LITELLM_PROXY_BASE_URL=http://127.0.0.1:4000 LITELLM_PROXY_KEY=sk-... \
  python -m capability_map_poc.eval_feedback --out-dir out/stage4a --model claude-sonnet-5-5
```

Stage 4a appends every label to `labels_<split>.jsonl` as it arrives and reuses them on the next run, so a rerun only pays for turns that are missing or failed
