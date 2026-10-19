# Shared proxy config

`Document::from_yaml` parses a config without filling defaults, resolving secrets or discarding extension fields. `Document::typed` projects it into `Config` for native consumers. `Config::from_yaml` and `Config::load` use the same document parser

The parser follows the existing Python loader's YAML 1.1 scalar behavior. For example, `on` is a boolean, `012` is octal, and `1e3` stays a string. Mapping order and duplicate-key replacement are preserved. Debug output excludes document values

`includes` owns the breadth-first include queue, duplicate detection, list extension and replacement of other values. Local paths resolve next to the declaring config, with the existing root-relative fallback and ambiguity warnings. File identity uses normalized absolute paths, matching Python rather than resolving symlinks

The Python bridge adapts this core behind the existing proxy and bucket loaders. Python retains source access, database and runtime settings precedence, secret resolution and lifecycle effects. Nested Python values pass through the include merge without serialization

Python falls back to PyYAML for aliases, merge keys, explicit tags, timestamps, binary values, sets, non-string mapping keys and integers outside the native numeric range. This preserves Python object identity, types and YAML exception details. Native parsing accepts nonrecursive aliases with bounded expansion, but cannot project every Python YAML value yet. A missing extension uses the existing Python compatibility implementation

This crate does not implement a second settings-precedence engine. Native request execution consumes typed configuration loaded at startup, without calling Python to parse it per request
