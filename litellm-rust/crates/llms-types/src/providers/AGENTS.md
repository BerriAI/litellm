# Provider data ownership

This directory owns shared provider-specific wire contracts and extensions, grouped by provider. Add format submodules when needed to separate distinct API contracts

Place a type here when its meaning belongs to the provider's protocol. Place it in `formats` when it belongs to the API format, even if only one provider supports it. A provider originating a format does not make that format's types provider-specific

Provider envelopes may reuse format contracts where their semantics match. Dependencies point from `providers` to `formats`, never the reverse. Do not copy a format model to add a provider field

A shared wire contract may live here independently of its current caller count. A decoding or rendering projection tailored to one transformation stays in `llms`. Serializability alone does not justify promoting adapter state or projections into public schemas

Parsing, formatting, shape validation, deduplication, and value semantics belong with the data. Authentication, required headers or betas, OAuth companions, header precedence, runtime defaults, capability checks, polling, retries, and normalization belong in provider adapters or the auth crates

An operation-status value belongs here. The decision to poll again belongs in the adapter. A provider header value belongs here. The decision to send it belongs in provider policy. Recognizing a wire value does not establish model support or equivalent behavior across providers

Follow the parent crate's contract-preservation rules. Test wire and value semantics here, including equality, ordering, and hashing for wire-equivalent header values. Test header selection, authentication, normalization, and execution in their owning crates
