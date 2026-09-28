# Accounting API

This first version implements a per-call settlement state machine with injected storage operations. It does not implement prices, admission policy, Redis operations, database writes, or Python callback dispatch. Production calls are not connected to it yet

## Ownership

Core supplies execution facts and reported usage. Accounting preserves those inputs and tracks settlement progress. The host drives asynchronous operations and owns cancellation cleanup. An accounting backend implements monetary budget reconciliation, spend recording, and release of remaining monetary reservations

The Python integration belongs at the `python-bridge` boundary. `host-python` provides runtime mechanics, and `callbacks-legacy-python` preserves callback delivery. No crate is renamed. Existing Python accounting remains active until a real adapter replaces its orchestration. That adapter must settle accounting before terminal callback fan-out, supply callbacks their accounting payloads, and preserve public entrypoints and ordering without giving callbacks ownership of settlement

## Rate limiting boundary

Rate limiting owns request and token windows, deployment quotas, and concurrency permits. It consumes execution facts and usage directly, independently of pricing or monetary settlement. It does not use accounting's backend, receipts, effects, or settlement status

Python currently has deployment rate checks in the router and key/user/concurrency limits in proxy hooks. `crates/router` is the Rust counterpart of `litellm.Router`, with deployment quota checks belonging near selection and retries. Axum's HTTP router mounts gateway routes and can inject request-level admission through middleware. SDK calls need an injected limiter without HTTP middleware. These implementations remain separate follow-up work

Prefer a library for the limiter algorithm. [governor](https://docs.rs/governor/latest/governor/) provides a transport-independent GCRA engine, already used for gateway UI login throttling. [axum-governor](https://docs.rs/axum-governor/latest/axum_governor/) adds HTTP middleware; [axum-limit](https://docs.rs/axum-limit/latest/axum_limit/) offers extractor integration and a Redis backend. Selection requires tests against LiteLLM's window, token reservation, usage reconciliation, and multi-node policies. None of these choices requires an accounting dependency

The host composes accounting and limiter cleanup around the same call lifecycle. It must attempt both even when either fails. Streaming reconciliation and permit release follow body completion, failure, or cancellation rather than response header delivery. The eventual lifecycle composition must preserve Python ordering while providing both services the shared facts they need

Coordination stores for budgets and rate limits must be injected independently of response-cache storage. Changing or disabling the response-cache backend must not replace either service's coordination dependency

## Inputs

`BudgetAdmission<R>` contains an optional, already-acquired monetary budget receipt. Receipt type `R` belongs to the backend and can identify a monetary reservation or retain its budget coordination context. Constructing a session does not enforce a budget or acquire a reservation. Rejected admission never creates an admitted session

`ReportedUsage<U>` distinguishes unknown usage from a reported value, including a reported zero. The consumer supplies its typed usage representation. Known cost does not imply known usage, and reported usage does not imply known pricing

`Usd` uses decimal-backed [rusty-money](https://docs.rs/rusty-money/latest/rusty_money/) amounts. `Charges::new` validates nonnegative USD components, and totals use the library’s checked arithmetic. Fractional cents remain intact without currency-minor-unit rounding. Callers supply decimal amounts rather than floating-point values. Signed reservation adjustments belong to the backend, not this charge type

`Charges` preserves provider and independently incurred service costs. Each component is known or unknown. An unknown component leaves the total unknown without erasing the known component

`Terminal<U>` records success, failure, or cancellation, reported usage, chargeable provider work, and charges. `ProviderWork::NotStarted` requires an established fact that this call incurred no provider generation work. It makes the provider charge zero while retaining usage and service charges. A response-cache hit can use this classification even when the cached response reports usage. Provider prompt-cache usage still represents started provider work

The inspected main branch has no shared provider/model/source/cache-hit execution-facts contract yet. Production wiring requires a core fact establishing whether this call started provider generation, retained reported usage on cache hits, and terminal facts for partial failure and cancellation. Reuse the cache work’s shared contract when it lands, without importing that branch or duplicating its provider/model/source/cache-key types here. Missing facts must not be classified as avoided provider work

Failure and cancellation never automatically zero charges. Pricing and admission policy must supply incurred costs, including partial-stream costs or an input-cost estimate where the established policy requires one. This API preserves those decisions rather than guessing charges from a terminal outcome

## Settlement

`finish` selects one immutable terminal value. A second delivery returns `AlreadyTerminal` and cannot replace the first value. `settle` requires a selected terminal and attempts pending effects in this order: monetary budget reconciliation, spend recording, remaining budget reservation release

Budget reconciliation and release run only when a monetary budget receipt exists. Spend recording runs even without a monetary reservation. This API has no response-cache dependency or callback registry

`Backend::apply` receives the terminal, admission receipts, and read-only progress of all effects. Its future is awaited inline and need not be `Send`, allowing a Python adapter to preserve the caller's task and context

| Backend result | Meaning | Session behavior |
| --- | --- | --- |
| `Accepted` | Responsibility transferred to a queue or another owner, not committed | Retain the acknowledgement and do not resubmit |
| `Committed` | The effect completed according to the backend's persistence contract | Retain completion and do not repeat it |
| `NotApplied(error)` | The effect made no changes and transferred no responsibility | Preserve the error; allow an explicit retry |
| `Indeterminate(error)` | The effect may have applied or transferred responsibility | Preserve uncertainty and forbid automatic retry |

Every effect records its result before the next effect starts. A failed effect does not skip later effects, including release. A backend must report partial application or a lost acknowledgement as indeterminate, never not-applied. Returning from an existing Python writer is insufficient evidence of commitment

Release must use the supplied progress to release remaining resources without undoing reconciled charges or independently refunding an uncertain adjustment. Release covers only monetary budget reservations. The backend must define how failed or uncertain monetary reservations are repaired; this crate does not invent that storage policy

`retry_not_applied` makes only the selected failed effect pending again. A backend must implement effects against stable admission identities and account for previously completed cleanup when explicitly retried. Accepted, committed, pending, interrupted, and indeterminate effects cannot be reset by this method

`SettlementStatus::Accepted` distinguishes queue acceptance from `Committed`. `Unpriced` means effects have been acknowledged but at least one cost remains unknown. `NeedsAttention` means an effect failed or its completion is uncertain. Per-effect progress remains available in every case

## Cancellation and recovery

The state becomes `InFlight` before invoking a backend. Dropping the settlement future leaves that effect in flight because its external result is unknown. A later `settle` attempts only the remaining pending effects, including release, and does not replay the interrupted operation

The host must retain the session and resume remaining cleanup through its cancellation-safe finalization path. Dropping the entire session performs no asynchronous cleanup. Cancellation during release itself leaves release uncertain and requires backend-specific recovery

This is local protection within one retained session. There is no persistent settlement key, remote deduplication, crash recovery, queue acknowledgement reconciliation, or automatic resolution of unknown prices. A new session can repeat the same remote operations. Persistent idempotency and recovery require a later backend implementation

## Validation

Public API tests inject backend outcomes at every effect, drop futures at every suspension boundary, and verify completed work is not replayed. They cover optional monetary admission receipts, unknown and zero values, partial usage on failure and cancellation, avoided provider work with service charges, overflow, queue acceptance, and release progress

The next Python adapter must consume shared execution facts and separate existing built-in settlement from CustomLogger delivery. Native gateway accounting later supplies pricing, monetary budget policy, storage operations, and persistent idempotency behind the same contract. Both paths must avoid duplicate cost calculation and spend updates

Python callback compatibility, real pricing, coordination-store behavior, and native gateway accounting require production adapters and their own integration tests. The API tests do not claim those paths are migrated or validated
