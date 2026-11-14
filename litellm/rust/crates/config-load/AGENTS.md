# Config load

- This crate holds the rules for loading a config and does no I/O. It has no `std::fs`, tokio runtime or PyO3
- Every read is a `HostCall` the host answers through its `Reply`. The gateway answers from `std::fs` in `gateway/src/config.rs`, and python-bridge answers by awaiting the caller's own Python reader in `python-bridge/src/config/`
- `Includes::step` runs the rules until the next host call or the result. It never waits on anything but a reply, so a host needs no runtime to drive it and Python can call it before it forks workers
- The host parses. A document arrives as a `serde_yaml_ng::Value` and the rules never read text, so PyYAML and `serde_yaml_ng` each keep their own scalar rules until one parser serves both hosts
- A value the rules only carry stays opaque. python-bridge hands a Python object YAML cannot hold over as a tagged value and puts the same object back in the result
- A file location is absolute. The host makes the root absolute, and the rules only normalize it lexically, the way `os.path.abspath` does, so a symlinked include is a different location from its target
- An object key follows `posixpath` on every platform. `location.rs` pins `normpath` to CPython's output
- A warning is a host call too, so it is surfaced in order and before a later failure
- The typed `Config` lives in `litellm-config`. This crate never depends on a host and no data crate depends on it
