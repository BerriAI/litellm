# LiteLLM for Rust

Use LiteLLM as a Rust library:

```sh
cargo add litellm
```

```rust
use litellm::LiteLlm;

let client = LiteLlm::new();
```

The default features also provide the LiteLLM gateway CLI:

```sh
cargo install litellm
litellm
```

Library consumers that do not need the gateway can disable its server dependencies:

```toml
[dependencies]
litellm = { version = "0.1", default-features = false }
```
