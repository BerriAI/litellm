OCR resolves provider-declared secret names through `SecretSource` and `ResolvedSecrets`; messages, chat, responses, and transcription still read process environment variables directly and should adopt the same seam

OCR provider requests use the shared `litellm-http` pool. AWS and Google secret-manager SDK clients keep their SDK transports, which do not yet inherit the pool's proxy, TLS, certificate, timeout, or observability configuration. Preserve those SDK transports and configure them equivalently instead of forcing them through reqwest
