# YAML

## Scope

- This crate exists so a LiteLLM config.yaml means the same thing to the Rust proxy as to the Python proxy, which reads it with PyYAML's `yaml.safe_load`. It ports only what changes the value a config hands to LiteLLM. It is not a general PyYAML port
- In scope, matching PyYAML exactly
  - Plain-scalar resolution that changes a value's type: YAML 1.1 booleans (`yes`, `on`), null, integers (`0755`, `0x1F`, `1_000`, `1:30`) and floats that need a `.`
  - Merge keys (`<<`), repeated keys (last value wins, first position kept), anchors and aliases, and a single document
  - The core tags for those values: `!!str`, `!!int`, `!!float`, `!!bool`, `!!null`, `!!seq`, `!!map`, `!!merge` and the non-specific `!`
- Out of scope, as text or refused
  - Timestamp-shaped scalars stay the text the user wrote. PyYAML turns them into `date` or `datetime`, which LiteLLM's string fields reject, so no working config depends on them
  - A bare `=` stays text, where PyYAML raises
  - Every other tag is refused: `!!timestamp`, `!!set`, `!!omap`, `!!pairs`, `!!binary`, `!!python/*` and custom tags such as `!Ref`
- Add a feature only for a config value LiteLLM actually reads, and move its cases from `differs_from_pyyaml_on_purpose` into the parity table when you do

## Layout

- saphyr-parser scans and parses. `compose.rs` ports PyYAML's `Composer` and which scalars the parser resolves implicitly, `resolve.rs` the implicit resolvers in scope, and `construct.rs` `SafeConstructor` for the tags in scope, including `flatten_mapping`
- The output is `serde_yaml_ng::Value`, so config types deserialize as before
- Resolver matchers are written by hand, one per PyYAML regular expression, with the expression in the doc comment. No regex
- saphyr-parser is pinned with `=`. A bump reruns `tests/pyyaml.rs` before it lands

## Tests

- Every expected value comes from running PyYAML at the version in `uv.lock`, never from reasoning
- `tests/pyyaml.rs` is a hand-kept table of named cases. Its doc comment has the command that prints a case's expected value. A deliberate difference goes in `differs_from_pyyaml_on_purpose` with what PyYAML returns

## Limits

- Deliberately stricter than PyYAML: nesting past `litellm_python_compat::MAX_DEPTH`, counting the levels an alias brings in, and aliases that expand past a million nodes fail, because PyYAML shares an aliased node and a `Value` copies it. `tests/limits.rs` pins both. Integers beyond 64 bits fail, since a `Value` number holds 64 bits
- Known gaps: PyYAML collapses keys Python considers equal (`1` and `True`), while a `Value` mapping keeps both. A repeated anchor name is an error in PyYAML and takes the latest definition here, since saphyr reports anchors only by id
