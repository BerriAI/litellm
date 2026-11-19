`src/google/` is Google login as google-auth does it, for any Google service. `src/vertex/` is what litellm's Python `VertexBase` adds on top for Vertex AI routes. Read their own AGENTS.md before adding code to either

A change to how a Google or Vertex AI login path behaves updates [PARITY.md](PARITY.md) in the same commit. A path that works only through code we added is Legacy, a path that worked in Python and fails here is Unsupported, and a path that works or fails as it did in Python is left out
