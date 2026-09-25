These are executable requirements for the legacy callback boundary. The named `rstest` rows in [custom_logger_contract.rs](../custom_logger_contract.rs) state the expected behavior independently of the current implementation

Run from the repository root with the project's Python dependencies installed in `.venv`:

```sh
PYO3_PYTHON="$PWD/.venv/bin/python" \
PYTHONPATH="$PWD:$(.venv/bin/python -c 'import site; print(":".join(site.getsitepackages()))')" \
cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-callbacks-legacy-python --test custom_logger_contract
```

Append a case name to run one requirement, or append `-- --list` to list the matrix. Omit `--test custom_logger_contract` to also run the crate's existing unit tests

All assertions and case tables are in Rust. `support.rs` drives `run_legacy_call` with the production Messages machine and a local HTTP recording server. Its small protocol host projects fixture arguments and returns a fixed public exception for upstream errors; it does not implement callback dispatch or transformations. Consequently these tests cover the callback contract with the real route, not the Python bridge's argument binding or error classification

`fixtures.py` contains actual `CustomLogger` and `CustomGuardrail` implementations, observations, and interpreter cleanup. It has no tests or expected outcomes. It uses the real Python shim, `Logging`, logging worker, and proxy release/cleanup functions. No installed native extension needs rebuilding: the callback adapter and route run from the Rust test executable

The integration executable has its own interpreter, separate from the unit tests that install fake Python modules. Cases serialize access to Python's global callback registries, restore those registries afterward, drain logging work, and join their executors. Model prices belong to a synthetic fixture model, the cost map is local, and the secret source is empty. No provider credentials or external API calls are needed

| Requirement | Rust test | Rows |
|---|---|---:|
| Sync/async success/failure, global/request registration, duplicate registration across both | `registered_loggers_receive_each_eligible_event_once` | 12 |
| Message mutation and tool replacement, no return/original kwargs/replacement kwargs, sharing with the next callback | `pre_request_edits_reach_the_next_callback_and_provider` | 6 |
| Body/header mutation versus replacing the callback envelope, identity, actual provider input | `pre_api_mutation_reaches_the_wire_but_envelope_replacement_does_not` | 8 |
| Blocking pre-hooks, ordinary logger errors, cancellation, original exception identity, no provider replay | `callback_failures_do_not_replay_the_provider_or_switch_outcomes` | 13 |
| Deployment response replacement reaches the caller and normalized logging payload | `deployment_response_replacement_is_returned_and_logged` | 1 |
| Guardrail rejection logs failure with the selected error and suppresses success | `a_guardrail_rejects_the_response_without_dispatching_success` | 1 |
| No release, accept, repeated accept, reject followed by accept | `deferred_success_obeys_the_proxy_release_decision` | 4 |
| Sync/async stream consumption and close, deferred proxy disconnect, no premature success, usage and bytes | `streamed_usage_is_logged_once_when_consumed_or_closed` | 5 |
| Caller task/thread, executor submission, logging-worker task, context propagation | `callbacks_run_in_the_required_execution_context` | 9 |

Private adapter invariants remain in the existing Rust unit tests rather than being duplicated through HTTP:

| Requirement | Existing test location |
|---|---|
| Caller keyword copy, retained value identity | `call.rs::tests` |
| Credential inheritance, explicit `None`, lazy credential access | `preparation.rs::tests` |
| Unknown arguments, deployment replacement, cancellation | `adapter.rs::deployment_hooks_tests` |
| Captured body/header roots, opaque values, aliasing, property-generated payload edits | `adapter.rs::payload_tests` and its adjacent property tests |
| Internal-call suppression, failure-family continuation, idempotent correlation-context restoration | `adapter.rs::terminal_tests` |
| At-most-once native release, reentrant release, queue failure, cancellation, collection of logger cycles | `deferred.rs::tests` |

This is a requirement map, not a claim to cover every `CustomLogger` method. Router/proxy hooks outside this boundary, vendor-specific transformations, and bridge input/error mapping belong to their owners. The HTTP stream fixture does not prescribe TCP chunk boundaries, so early-close cases assert logging of delivered input usage without assuming how many SSE events fit in the first read

Add a row when an existing operation has a new requirement. Add a focused table when the observed behavior differs. Keep expected events and values explicit. Do not generate expectations from the implementation, use Python as an automatic oracle, or add a broad scenario language

A callback can intentionally mutate caller objects, so equality and identity are separate assertions. Sequential hooks have exact event ordering. Context tests distinguish caller, executor, and logging-worker delivery without requiring an incidental ordering between independently scheduled background tasks

The initial baseline has 53 passing and six failing integration cases, with no ignored cases. The six failures encode three outstanding requirements: pre-request message edits must reach the provider (three rows), proxy acceptance must release native Messages success logging (two rows), and deferred logging must survive proxy disconnect cleanup (one row). Fix production behavior in a later change, keeping these assertions intact

Three targeted mutation checks passed: skipping pre-API dispatch, ignoring returned request kwargs, and discarding deployment response replacement each made its previously passing case fail. These checks establish sensitivity for those requirements, not an overall mutation-coverage score

The Python fixture passes Ruff. A standalone strict Basedpyright run reports diagnostics at the legacy boundary, including untyped callback constructors and registries, protected proxy helpers, and dynamically attached logging attributes. This fixture is outside the repository's configured Python type-check scope; no type-check budgets or suppressions were changed
