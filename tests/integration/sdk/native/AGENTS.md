This directory holds only the tests that cannot be written in the Rust code

Every file here needs the compiled `litellm.rust_bridge._native` extension: the conftest imports it at collection and marks every test `requires_rust_extension`. These files are GitHub-only (`GITHUB_FILES` in `tests/integration/run.py`) and run through `make test-rust-extension`
