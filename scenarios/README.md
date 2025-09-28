# LiteLLM Scenarios

The scripts in this directory exercise end-to-end flows that require real LLM
providers or auxiliary services. They are **not** part of the automated unit
suite—run them manually when you want to validate integrations.

## Usage

* Set the provider credentials required by the scenario (see each module for
  details).
* Execute the script with the Python interpreter in the repo virtualenv, e.g.
  `python scenarios/mini_agent_local.py`.
* Scenarios exit with a helpful message if required environment variables are
  missing.

## Available Scenarios

* `mini_agent_local.py` – Calls the mini-agent FastAPI shim using the local
  backend (no external providers required).
* `codex_agent_router.py` – Routes a request through the `codex-agent` provider
  (requires the codex-agent shim or remote endpoint).
* `parallel_acompletions_demo.py` – Runs `Router.parallel_acompletions` against a
  live provider to illustrate fan-out behaviour.
