# litellm/_auth

## Why this exists

- Lots of Python code like `litellm/proxy/auth/user_api_key_auth.py` is too complex
- Soon, Rust will implement the same auth stack, and expose common functionality as Python API through pyo3, to avoid maintaining 2 copy of code.
- So the goal of `_auth` here is to define that interface first, not writing really good python code. Should be behavior preserving refactor though
