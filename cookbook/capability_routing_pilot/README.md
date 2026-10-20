# Capability-aware routing pilot (standalone research POC)

Five frozen SWE-bench dev tasks (`tasks.json`) are solved by mini-swe-agent 2.4.6 with Claude Haiku 4.5, Sonnet 5, and Opus 5 through a LiteLLM gateway, then graded only by the official swebench 5.0.2 harness. This is not production code and nothing under `litellm/` imports it

`poc/guarded_model.py` wraps the mini-swe-agent LiteLLM model and refuses a call before dispatch when spent plus a worst-case reservation would exceed the per-run budget. Actual cost comes from the gateway response cost header. Missing, non-finite, or negative costs fail closed, mark accounting incomplete, and stop further dispatch. New dispatch stops below 50 GiB free. A running solver process group is killed if free space falls below 20 GiB, with a 10-second poll interval and a 600-second timeout that returns rc 124. Only a first infrastructure failure with zero model-call rows and zero cost is eligible for one retry. Emergency disk stops and accounting failures are never retried. `poc/run_pilot.py` runs the 15-run matrix and appends one record per attempt to `runs.jsonl`. `poc/label.py` writes offline GPT-5.4-mini labels and runtime Jev labels from the problem statement only. `poc/policy.py` builds the capability map and the conservative router, and `poc/analyze.py` writes `capability_map.json` and the report

The router picks a cheaper model only when, on the shared graded task IDs, a one-sided Hoeffding lower bound on the paired success difference is at least minus the tolerance and the paired mean cost is lower. With five tasks it falls back to the Opus reference, by design

Three evaluator images failed gold validation with an unchanged grader, NumPy 2 in both pvlib images and pytest 8 in the pydicom image. `overlays/` holds the version pins applied locally under the original tags. Gold then resolved 5/5 and a no-op patch 0/5. `images_pinned_upstream.txt` records upstream registry manifest digests, while `images_pinned.txt` separates those from local image IDs. `selection_candidates.json` preserves historical selection provenance

## Reproduction

Run these commands from this directory. `pip_freeze.txt` is authoritative for the environment

```sh
uv venv
uv pip install -r pip_freeze.txt
source .venv/bin/activate
```

Pull each upstream image by its registry manifest digest, tag it locally, and inspect the recorded repository digests. Each inspection output must contain the listed `image@sha256:<upstream digest>` entry before building overlays

```sh
docker pull swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1395@sha256:c7b51f4978926d54a788c54ac4d96e6c4c4e2cd2ee62736b30e91c9fd0891f10
docker tag swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1395@sha256:c7b51f4978926d54a788c54ac4d96e6c4c4e2cd2ee62736b30e91c9fd0891f10 swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1395:latest
docker image inspect --format '{{json .RepoDigests}}' swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1395:latest

docker pull swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-884@sha256:ca28dab1d331c1c7e4908784576a7e1f3b11387dbaa4a8db466ce42a0eda6bec
docker tag swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-884@sha256:ca28dab1d331c1c7e4908784576a7e1f3b11387dbaa4a8db466ce42a0eda6bec swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-884:latest
docker image inspect --format '{{json .RepoDigests}}' swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-884:latest

docker pull swebench/sweb.eval.x86_64.pylint-dev_1776_astroid-1614@sha256:7206ea531b6c8b02f7a8e1760d732e4651f95307781f12f55e4b88e982610159
docker tag swebench/sweb.eval.x86_64.pylint-dev_1776_astroid-1614@sha256:7206ea531b6c8b02f7a8e1760d732e4651f95307781f12f55e4b88e982610159 swebench/sweb.eval.x86_64.pylint-dev_1776_astroid-1614:latest
docker image inspect --format '{{json .RepoDigests}}' swebench/sweb.eval.x86_64.pylint-dev_1776_astroid-1614:latest

docker pull swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1165@sha256:71e033964894077384c0c27a04ae0271e28eaad35602aa7570d40c683f1967d1
docker tag swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1165@sha256:71e033964894077384c0c27a04ae0271e28eaad35602aa7570d40c683f1967d1 swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1165:latest
docker image inspect --format '{{json .RepoDigests}}' swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1165:latest

docker pull swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1031@sha256:0b962e881760f4b8bf3cdbfc695624c90906198ccf1e2410cd639f6adc6e340b
docker tag swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1031@sha256:0b962e881760f4b8bf3cdbfc695624c90906198ccf1e2410cd639f6adc6e340b swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1031:latest
docker image inspect --format '{{json .RepoDigests}}' swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1031:latest
```

```sh
docker tag <img> routing_poc/orig.<name>:latest
docker build -f overlays/Dockerfile.<name> -t <img> .
```

For the three overlays, preserve the upstream tag first, then build the local image under that tag. These local builds have image IDs but no upstream registry manifest digest

```sh
docker tag swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1395:latest routing_poc/orig.pvlib_1776_pvlib-python-1395:latest
docker build -f overlays/Dockerfile.pvlib_1776_pvlib-python-1395 -t swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1395:latest .
docker tag swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1165:latest routing_poc/orig.pvlib_1776_pvlib-python-1165:latest
docker build -f overlays/Dockerfile.pvlib_1776_pvlib-python-1165 -t swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1165:latest .
docker tag swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1031:latest routing_poc/orig.pydicom_1776_pydicom-1031:latest
docker build -f overlays/Dockerfile.pydicom_1776_pydicom-1031 -t swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1031:latest .
```

Run gold and no-op evaluator controls for the five selected task IDs

```sh
python -m swebench.harness.run_evaluation -d SWE-bench/SWE-bench -s dev -p gold -i <5 ids> --max_workers 3 -id gold_check -t 900
python -m swebench.harness.run_evaluation -d SWE-bench/SWE-bench -s dev -p noop_preds.json -i <5 ids> --max_workers 3 -id noop_control -t 900
```

The no-op patch adds the empty-ish file `routing_poc_noop.txt`

Prepare or verify the pinned dataset

```sh
python -m poc.prepare
python -m poc.prepare --check
```

The original selection rule sorted SWE-bench dev IDs by `sha256(seed + instance_id)` using seed `litellm-capability-routing-pilot-v1`, then took the first five with a public evaluation image. `selection_candidates.json` records that historical provenance only. Reproduction uses the frozen IDs and order in `tasks.json`, loads the pinned dev revision, and projects only the six agent fields into `agent_dataset/train.jsonl`. It does not reselect tasks, query Docker Hub, or write `tasks.json`

Create a gateway virtual key scoped to the five models below with `max_budget` 15. Store the key in `.private/pilot_key` with mode `0600`, then export the gateway URL. Never commit or print the key

```sh
install -d -m 700 .private
chmod 600 .private/pilot_key
export LITELLM_GATEWAY_BASE_URL=https://gateway.litellm-sandbox.ai
```

The five allowed models are `anthropic/claude-haiku-4-5-20251001`, `anthropic/claude-sonnet-5`, `anthropic/claude-opus-5`, `openai/gpt-5.4-mini-2026-03-17`, and `typesafe/jev-latest`

Run labeling, the pilot matrix, analysis, and offline tests

```sh
python -m poc.label
PYTHONPATH=. python -m poc.run_pilot --workers 3
python -m poc.analyze
python -m pytest -q tests
```
