# Migrating built-in callbacks to the v1 contract

How a built-in logger moves from `CustomLogger` (`litellm/integrations/`) to the v1
contract (`litellm/callbacks_v1/__init__.py`), and what the scaffolding in `builtin/` is
for while that happens. The rules for writing one port are in `builtin/AGENTS.md`; this
file is the flow around them.

## The shape of the migration

- Ports live in a parallel tree, `builtin/<name>.py`, one frozen class each declaring
  `SinkPort` or `InterceptorPort` from `builtin/port.py` as its base. The legacy logger
  is never edited, renamed, or taught about its port.
- Everything the migration needs to say is in `builtin/manifest.py`: `Gap`, each port's
  registration name, its twin, its status and its gaps. A port carries none of it, so
  cutting over deletes that file and leaves the ports untouched.
- A port is inert. Importing it registers nothing and the product never registers it;
  only tests do. Writing a port therefore cannot change behaviour for anyone.
- A port goes live by removing its legacy twin. The two are never active on the same
  call, so there is no dual delivery, no ownership table, and no per-call skip logic.
- v1 envelopes only exist on native (Rust) routes. A port can only replace its twin once
  every route the twin serves emits them.
- Built-in and custom callbacks share one interface, the v1 contract. In the SDK a custom
  callback is free-form Python, in process, never sandboxed. On the gateway it runs in an
  out-of-process extension host driven by the same `CallSession` (`litellm-rust/crates/
  callbacks-v1`). A built-in port is pure methods over values with every effect in
  `builtin/runtime.py`, which also owns the two subscribers (`Sink`, `Patcher`); that is
  hygiene for built-ins (secrets, a later Rust port), not the isolation mechanism.

Porting comes first, and on purpose, because its first product is not the port. It is the
list of things the contract cannot do yet. The contract grows from what real built-ins
needed, not from guesses.

## The loop

```
  port a built-in ──► record gaps ──► read the backlog ──► grow the contract
        ▲                                                        │
        └──────── delete gaps and their allowances ◄─────────────┘
```

### 1. Port inertly

Write `payload()` against the golden envelopes. Every time the twin reads something the
envelopes do not carry, do one of two things and write a `Gap` either way:

- leave the field out of the vendor body, or
- let `builtin/runtime.py` work around it for every port at once (`CallJoin` recovers the
  model for the terminal envelope; `Outbox` keeps vendor I/O off the call path). A
  workaround never goes inside a port: a port that needs a thread, a socket or state has
  found a `capability` gap.

A `Gap` is handwritten, in `manifest.py` beside the port's entry. It has a `kind`
(`fact`, `event`, `capability`), a `key`, the `legacy` input the twin read, and a `note`. The `key` names what the contract would have
to grow (`cost`, `usage`, `identity`), not what the twin happened to call it, so that two
ports missing the same thing say so in the same word.

Then write the parity test: one real native call, the twin and the port each posting to
their own recording vendor, `assert_parity` on the two bodies. Every difference needs an
`Allowed(path, reason)`. If the difference exists because of a gap, it cites the gap:
`Allowed("data.cost", "no cost fact on call.succeeded", gap="cost")`. An `Allowed` with no
`gap` is a difference the port means to keep (each side stamps its own clock, the port
omits a key the twin sent as null).

Add the port to `manifest.py` as `exploring`, then `parity` once the parity test passes.

### 2. Read the backlog

```bash
python -m litellm.callbacks_v1.builtin.manifest
```

prints the ledger (every gap, per port) and the backlog (every gap key, with the ports
waiting on it, most wanted first). A key several ports wait on is contract work. A key
one port waits on is either contract work or a candidate for "v1 deliberately does not
do this"; decide which before building it.

### 3. Grow the contract

The contract is defined in Rust (`litellm-rust/crates/callbacks-v1`, golden envelopes in
`golden/v1/`) and mirrored by the types in `litellm/callbacks_v1/__init__.py`. Add the
fact, event, or capability there, for every native route, and update the golden files.

### 4. Close the gap

For each port that listed the key:

1. Use the new fact in `payload()`, or drop the workaround.
2. Delete the `Gap`.
3. Run the parity tests. They now fail in exactly the places that need attention:
   - an `Allowed` still citing the deleted key fails with "cites gap ..., which the port
     does not list";
   - an `Allowed` whose difference is gone fails as stale;
   - a difference the new fact did not actually fix fails as unexplained once its
     `Allowed` is removed.
4. Delete those `Allowed` entries until the test is green again. Do not re-allow a
   difference to get there.

The order of 1 and 2 does not matter; the tests fail until both are done.

### 5. Cut over

A port is `ready` when its parity test passes, its manifest gaps are empty, and every route its twin
serves emits v1 envelopes. `test_conventions.py` refuses `ready` with an open gap. A
`ready` port replaces its twin: register the port where the twin was registered, remove
the twin, in one change.

## What is checked and what is trusted

| Claim | Checked by |
|---|---|
| Every difference from the twin is explained | `assert_parity`: an uncovered difference fails |
| The allow-list does not outlive the differences | `assert_parity`: a stale `Allowed` fails |
| The allow-list does not outlive the gaps | `assert_parity`: an `Allowed` citing an unlisted key fails |
| A port is not `ready` with open gaps | `test_conventions.py` |
| The gap list is complete | nobody |

The last row is the honest one. The gap list is only as complete as its author. For a
sink, the parity diff is the real source of truth and the gaps are its readable summary: anything the
author forgot still shows up as an unexplained difference. Three kinds of gap never show
up in a diff, and the handwritten entry is their only record:

- `capability` and `event` gaps (observers run inline; an interceptor cannot see
  `call_type`), because they are about how the port runs, not what it sends;
- gaps the port works around (`terminal_model`), because the workaround makes the bodies
  equal;
- gaps the parity call does not exercise (`request_user`, when the test sends no `user`).

Closing one of these is a manual edit with no failing test to prompt it. When the contract
change lands, grep the key.

## Open decisions

Porting does not settle these, and no port may settle one by accident. Each is kept out of
the ports so that it can be decided once, in the host.

- **Where `Config` comes from.** A port takes a finished `Config`. Whether the host
  resolves it from the twin's environment variables, from config.yaml, or through the
  Rust settings precedence is undecided; `from_env` on a port only records the names.
- **The gateway's extension host.** Custom Python callbacks run out of process there:
  the worker's confinement profile, the interceptor timeout and failure policy, and
  whether built-ins run in a trusted worker or move to Rust. The interface is the v1
  Protocol itself over a socket, not the ports' pure-method shape.
- **Who holds secrets.** A `Config` carries the vendor's API key and a `Delivery` carries
  it as a header, so a port sees the credential. A sandboxed callback should name a secret
  and let the host put it on the wire. `off_path_delivery` is the gap that would carry it.

## Differences that are never closed

Some legacy inputs will not come to v1 (`hidden_params` is the likely example). Those are
not gaps; a gap is something the contract intends to grow. When that decision is made for
a key, delete the `Gap`, drop `gap=` from its `Allowed` entries and reword their reasons
as the decision, and move the feature to the port's `Not ported yet:` docstring line or to
the release notes. A port must not sit at `parity` forever behind a gap nobody will close.

`parity` means every difference is explained. It does not mean the port is equivalent:
`generic_api` is at `parity` while shipping about a fifth of `StandardLoggingPayload`.
Only `ready` means equivalent.

## What survives the migration

| Piece | After every twin is gone |
|---|---|
| The ports, `SinkPort`, `InterceptorPort`, `Delivery`, `CallRecord`, `Batching` | stay; this is the product and its interface |
| `runtime.py`: `Sink`, `Outbox`, `Transport` / `HttpxTransport` | stays until the contract delivers off the call path itself (`off_path_delivery`); then the host owns it |
| `CallJoin` | stays only if the terminal envelope never carries model and metadata; it is a workaround for the `terminal_model` gap |
| `manifest.py` entire: `Gap`, every gap list, the names, `ledger()`, `backlog()`, the statuses, `Entry.legacy` | deleted with the last twin |
| The twin checks in `test_conventions.py` | deleted with the last twin |
| `assert_parity`, `Allowed`, `test_parity.py` | deleted; there is no twin left to compare with. The payload tests on golden envelopes remain as the ports' tests |

At that point `port.py` holds the two protocols and what they are written in, which is the
whole of what a callback author, built-in or custom, needs besides the contract types.
