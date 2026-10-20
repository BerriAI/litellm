# Capability-aware routing pilot (standalone research POC)

Five frozen SWE-bench dev tasks (`tasks.json`) solved by mini-swe-agent 2.4.6 with Claude Haiku 4.5, Sonnet 5 and Opus 5 through a LiteLLM gateway, graded only by the official swebench 5.0.2 harness. This is not production code and nothing under `litellm/` imports it

`poc/guarded_model.py` wraps the mini-swe-agent LiteLLM model and refuses a call before dispatch when spent plus a worst-case reservation would exceed the per-run budget. Actual cost comes from the gateway response cost header. `poc/run_pilot.py` runs the 15-run matrix with one setup-only retry for infrastructure failures and appends one record per attempt to `runs.jsonl`. `poc/label.py` writes offline GPT-5.4-mini labels and runtime Jev labels from the problem statement only. `poc/policy.py` builds the capability map and the conservative router, and `poc/analyze.py` writes `capability_map.json` and the report

The router picks a cheaper model only when, on the shared graded task IDs, a one-sided Hoeffding lower bound on the paired success difference is at least minus the tolerance and the paired mean cost is lower. With five tasks it always falls back to the Opus reference, by design

Three evaluator images failed gold validation with an unchanged grader (NumPy 2 in both pvlib images, pytest 8 in the pydicom image). `overlays/` holds the version pins applied locally under the original tags. Gold then resolved 5/5 and a no-op patch 0/5. `images_pinned.txt` lists the digests used and `images_pinned_upstream.txt` the upstream ones

Run tests with `python -m pytest -q tests` from this directory with mini-swe-agent, litellm, scipy and pyyaml installed
