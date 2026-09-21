# Python interop foundation PR plan

Proposed implementation PR title: `fix(rust): preserve Python settings semantics at the native boundary`

Base: `main` at `457b01e96d131f88df8cace8832f8e044ee5f167`. Planning branch: `litellm_python_interop_foundation`

This plan follows `migration/tdd/sections/rust/python-interop/{pyo3-contract,boundary,coercion}.typ` in the sibling `litellm-typst` checkout. The first PR establishes conversion contracts and exercises them through existing HTTP, URL policy, and OCR provider-default consumers. The foundation is implemented on this branch. No PR is being created for this task

**Baseline before implementation**

`host-python/src/marshal.rs` already uses `pythonize` directly, separates internal conversion from public argument errors, and contains serializer panics in `Pythonized<T>`. Keep those entrypoints. `Pythonized<T>` currently stringifies conversion errors, unlike `from_py` and `to_py`, so its error transfer needs correction

`core-utils/src/serde_compat.rs` already provides composable `LaxI64` and `FiniteF64` adapters. They first deserialize into `serde_json::Value`. Preserve their accepted input contracts while moving scalar decoding to Serde visitors, avoiding an intermediate JSON representation and making behavior through `pythonize` explicit

`python-bridge/src/http.rs` currently uses strict derived Boolean and string extraction for mutable settings, converts extraction errors into `RustBridgeDeclined`, and silently drops unsupported `ssl_verify` values. OCR provider defaults have the same strict-extraction/decline pattern. `python_settings.json` checks field names only. These are the initial production consumers and regressions for the foundation

**Ownership and placement**

Keep interpreter attachment, structured-data conversion, panic containment, and execution adapters in `host-python`. Keep field paths, configuration error policy, named settings coercion, and product-specific tagged inputs in `python-bridge`. Keep pure parsing and Serde adapters in `core-utils`. This requires no new crate and no domain dependency from `host-python` into `core-utils`

Add one focused `python-bridge/src/coercion.rs` module containing the field reader, semantic wrappers, and projection errors. Use `core-utils/src/serde_compat.rs` for shared token/numeric parsing and Serde decoding initially, splitting only if its size warrants it. The bridge can call those pure helpers through its existing dependency. Do not introduce a generic coercion registry, runtime manifest dispatch, or a new conversion framework

Settings snapshots hold raw `Bound<'py, PyAny>` values only during attached projection. Successful adapters return owned values. Python snapshot dataclass annotations use `object` for arbitrary mutable globals, retaining strict annotations for accessor-owned fields such as `user_agent` and `readable`. No live settings object, iterator, or borrowed Python value enters native HTTP state

**Serde and structured conversion**

Keep `from_py` and `to_py` as the internal, exception-preserving conversion path. Make `Pythonized<T>` use the same standard `PythonizeError` to `PyErr` conversion without losing its panic guard. Keep the explicitly argument-focused `ValueError` contract of `from_py_argument`; settings projection never passes through that helper

Implement scalar visitors behind the existing `LaxI64` and `FiniteF64` adapter names. Preserve integer precision beyond the exact f64 range, signed bounds, supported decimal/underscore strings, Boolean numeric behavior, fractional rejection for integers, and nonfinite rejection for floats. Preserve composition inside `Option` and sequences, missing/null behavior at the field boundary, and ordinary numeric serialization. Use the existing [serde_with DeserializeAs contract](https://docs.rs/serde_with/3.16.1/serde_with/trait.DeserializeAs.html)

Add the pure `parse_str_bool(&str) -> Option<bool>` parser used by bridge `StrBool`, HTTP `SslVerify::parse`, and the environment switch helper. It recognizes trimmed, case-insensitive true/false only. The environment helper retains its separate rule that only `Some(true)` contributes an enabled layer. Numeric helpers are shared only where a second actual consumer needs them

Run common ordinary-value fixtures through JSON deserialization and direct Python-to-typed-Serde conversion. Restrict equivalence claims to their overlapping input domain. Live descriptors, identity, truthiness, iteration, and stringification are exercised through PyO3 separately. Do not add unused Serde counterparts for Python-only semantics or convert live settings through JSON text, `serde_json::Value`, `repr`, or `py_literal`

**Named settings semantics**

| Adapter | Contract and first consumer |
| --- | --- |
| `Truthy` | Execute Python truth testing and preserve its exception; IPv4, URL validation, and trust-env globals |
| `ExactTrue` | Compare identity with the True singleton without equality or truth testing; HTTP2, transport disable, and token refresh |
| `StrBool` | None or an actual string parsed by the shared parser; no arbitrary stringification |
| `OptionalStrictString` | None or an actual string, including the empty string; client certificate |
| `FalsyOptionalString` | Test truthiness first, treat falsey as absent, reject truthy non-strings; provider project and location |
| `TuningString` | Test truthiness first, then retain actual strings and ignore other values; TLS tuning |
| `StringCollection` | Apply the field's explicit container/member policy and return owned strings; URL allowlist |
| `SslVerifyInput` | Classify None, actual Boolean, Boolean string, CA path, live SSLContext, and invalid types separately |

A single generic Boolean or optional-string coercer cannot implement these contracts. Accept string subclasses by their Unicode contents without calling overridden convenience methods. Use no `.ok()` or default value to discard an error from a Python protocol operation

For URL hosts, a direct string represents one host. Otherwise test container truthiness and iterate, test member truthiness, skip falsey members, and reject truthy non-string members. Normalize with the existing URL-policy rules, deduplicate, and sort only where membership makes order irrelevant. Keep origin/scheme/port parsing in its existing domain helper rather than applying hostname normalization blindly to arbitrary URL strings

**Errors and first production adoption**

Represent projection failures as a tagged result separating original Python exceptions, invalid configuration, unsupported live objects, and internal accessor/schema failures. Map it once at the bridge: preserve original `PyErr`, use field-focused `ValueError` for invalid or unsupported configuration, and `RuntimeError` for genuine internal schema failures. Diagnostics contain group, field, expected forms, and actual type, never the supplied value or its representation

Attribute, truthiness, iteration, and explicit stringification exceptions retain identity, traceback, cause, and context. In particular, do not relabel an AttributeError deliberately raised by a descriptor as a missing-field schema failure. Contract validation must distinguish schema drift from errors executing Python behavior

Convert HTTP and URL snapshots, per-call `ssl_verify`, and OCR provider defaults through the named adapters. Preserve existing call/environment/global precedence and projection timing. Finish projection before constructing the native client or starting provider I/O. Invalid values and live SSLContext must raise configuration errors rather than disappear or authorize fallback

Keep certificate paths through projection and validate empty, missing, or unusable client-certificate paths before I/O. The current native layer filters empty client-certificate paths, so correcting that narrow downstream behavior is part of adoption. Keep the existing CA-bundle missing-file policy explicit and separately tested; do not silently conflate it with client-certificate validation

Classify the existing Secret Manager `readable` field as a strict accessor Boolean. Full Secret Manager client/system/settings snapshots and callback execution remain a separate PR. The existing readable-manager capability gap must be documented and must not be reported as fixed by this foundation

**Semantic manifest**

Extend `python_settings.json` with a stable group version and field records containing adapter ID, requiredness, precedence role, sensitivity, and specialized accepted/unsupported shapes. Include only the snapshot fields that exist in this PR. Update the Python contract test and a static Rust `SettingSpec` table to agree with the manifest

The manifest checks declared contracts; behavioral tests prove the adapters implement them. Projection stays direct typed code. A manifest row alone is never evidence that a coercion works

**Behavioral validation**

Extend the existing mapped Python settings tests and Rust marshal/HTTP tests. A new coercion module may have its own focused Rust tests. Use the existing installed-extension OCR suites for public-route regressions. Do not add source-text assertions or class-attribute monkeypatching

The acceptance matrix covers None, Boolean values, integer zero/one, strings, containers, subclasses, and arbitrary objects. Protocol fixtures raise pre-created exceptions from descriptors, `__bool__`, `__len__`, `__iter__`, and `__next__`; assert identity and exception chains. Verify ExactTrue never invokes hostile equality/truthiness. Verify falsey provider defaults preserve fallback, HTTP2 does not accept integer one, and a false/unknown environment token cannot switch off a true global

Cover a real SSLContext, unsupported objects, Boolean strings, certificate path failures, direct-string hosts, sets, generators, duplicates, mixed members, and protocol failures. Mutate globals and source collections between calls: the next snapshot observes changes and an already projected value stays unchanged. At the installed public OCR boundary, projection failures must cause zero provider requests and zero Python fallback calls under required-native execution

Run focused crate tests first, then the workspace Rust checks used by CI, `make test-rust-extension`, relevant Python settings tests, and `make check`. Use the fresh installed wheel and verify native provenance. Review the saved `make check` log rather than rerunning it to inspect output

Target mutation tests at truthiness versus strict extraction, identity versus equality, swallowed versus preserved exceptions, string-as-one versus character iteration, normalization/deduplication, numeric bounds, and terminal errors versus fallback. Aim for more than 90% killed non-equivalent mutants in the changed coercion paths

Before opening the implementation PR for maintainer review, provide a reproducible localhost proxy curl request with a real provider and positive native-execution evidence. Record the configured settings and user-visible result without credentials. Unit tests belong in validation, not the proof-of-fix section. Require the current tip's CI and coverage, Greptile confidence of at least 4/5, and acceptable Veria/Bugbot results; pending or unavailable results remain explicitly unresolved

**Commit sequence and follow-ups**

Start with the Serde visitor/error-transfer changes and their regression tests. Follow with field adapters and the shared token parser. Adopt them in HTTP, URL policy, and provider defaults together with the semantic manifest and installed-extension regressions. Keep these as reviewable commits in one foundational implementation PR

Follow-up PRs can add `OptionalRedisBool`, cache-specific stringification and collection rules, and full Secret Manager bindings using the same field/error machinery. Redis accepts a different token set from StrBool, so do not share their Boolean semantics. Runtime redesign, callback lifecycle changes, free-threaded support, wholesale request serialization, and unrelated cache work are outside this PR
