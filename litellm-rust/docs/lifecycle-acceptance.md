# Native lifecycle acceptance

This is an implementation checkpoint, not sign-off on the complete lifecycle migration

The behavioral references are the current Python implementations in `litellm/rust_bridge`, `litellm/litellm_core_utils/litellm_logging.py`, `litellm/proxy/common_request_processing.py`, and the Anthropic Messages streaming iterator. The callback design notes under `Rust Native SDK/callback` supply the ownership and compatibility requirements

| Area | Implemented foundation | Remaining acceptance |
| --- | --- | --- |
| Messages buffered | Route-specific preparation/send permits; single-use operation tickets; private raw executor | Carry one completion/accounting owner through Python setup, preparation, provider execution, and terminal operations |
| Messages streaming | Registered completion handle; synchronous core terminal recording; cancellation/drop cleanup; incomplete drain-capacity outcome; legacy Python drain logging | Core deferred completion must follow the real proxy decision; prove every handoff and disconnect boundary |
| Chat Completions | Route-specific permits; prepared execution no longer creates a nested lifecycle; native usage captured before response replacement/rejection | Remove remaining public signing helpers; bind preparation and provider commitment to the same accounting owner |
| OCR | Route-specific permits; prepared requests cannot enter the public executor without a send permit; no nested prepared lifecycle | Carry completion ownership across credential resolution and Python policy phases; integrate deferred proxy decision |
| Audio Transcription | Shared native preparation/runner and cancellation guard | Complete the same admitted/resumable owner migration and route-specific boundary tests |
| Realtime | Shared WebSocket completion established before dialing; cancellation recorded independently; observed usage retained | Separate logical identity from provider session identity; deterministic dial cancellation and warmed-transfer accounting gates |
| Responses WebSocket | Shared WebSocket completion established before dialing; handshake errors return terminal records; usage provenance | Deterministic dial cancellation, incomplete protocol outcomes, and logical/attempt accounting gates |
| Deferred result | Single-use accept/reject/drop capability; `into_result` explicitly accepts | Wire the capability through production Python and the proxy acceptance/rejection path; acceptance must release delivery as well as accounting |

`Operation::contract` now determines scheduling, replacement handling, and failure disposition. Python continues invoking direct and inline-awaited callbacks in its caller context. The bridge retains Python objects and exceptions; the core receives outcomes and owned native values

`CallLifecycle::issue` permits one outstanding operation. A ticket identifies its execution, generation, and operation. Completion rejects foreign or stale tickets. Preparation and provider permits have private constructors, are route-specific, and can only be issued through the sealed route lifecycle API. Compile-fail doctests cover the raw executor and permit construction boundaries

`BeforeProvider` describes transport commitment, not permission to replay through Python. Existing Python admission/fallback guards remain authoritative. Commitment still needs transport-level response receipt and normalization observations; a successful decoded response is not an adequate substitute for receipt of HTTP headers

Buffered cancellation records a core terminal without invoking ordinary Python failure callbacks. HTTP streaming records synchronously when the observed stream terminates, independently of callback-task scheduling. Runtime teardown drops native work and records cancellation even if the runtime cannot deliver callbacks. Repeated polling, close, drop, and completion-handle release cannot record the same observed stream twice

The native usage snapshot precedes deployment-success replacement. Post-response rejection and cancellation retain that snapshot. WebSocket snapshots distinguish unavailable usage from observed partial usage and promote it on successful session completion. Decoder failures before usage extraction still need route-specific accounting observations

The installed-wheel differential suite passed 291 selected tests at this checkpoint, including real integration implementations with controlled dependencies. The cancellation regression now runs on both Python and Rust: once the provider finishes a detached drain successfully, Python's source requires success billing even though the consumer was cancelled. The test uses a first-chunk signal and releases the provider explicitly instead of establishing ordering with a sleep

The transition tests use independently written expected traces. The focused mutation campaign covers operation identity, single provider issuance, completion transfer, cancellation recording, terminal deduplication, trusted usage, captured body bytes, drain-capacity classification, and callback sequencing. All nine mutations were killed after strengthening the same-phase foreign-ticket regression. This 100% result is a bounded regression score, not a claim of repository-wide mutation coverage

Completion of the original plan still requires every item in the remaining-acceptance column, complete host-accessible preparation/settlement/signing encapsulation, production deferred-capability wiring, and the requested exhaustive transport/cancellation and integration schedules. No new provider, endpoint, retry engine, or cache engine is introduced by this checkpoint
