# Python forwarding compatibility

The opt-in forwarding gateway serves HTTP/1.1 and WebSockets in front of Python. Python still owns every route and policy. Its protocol contract and environment settings are in [AGENTS.md](AGENTS.md)

## URL fragments and literal hashes

The gateway follows Hyper's `http::Uri` parsing. If a raw client sends an unescaped `#` in the request target, only the path and query before the first `#` reach Python. For example, `/v1/files?purpose=batch#retry&limit=1` becomes `/v1/files?purpose=batch`, and `/gemini/v1beta/files/a#b?key=k` becomes `/gemini/v1beta/files/a`

This is accepted behavior because `#` starts a URL fragment, which is separated before dereferencing ([RFC 3986 section 3.5](https://www.rfc-editor.org/rfc/rfc3986.html#section-3.5)). HTTP origin-form contains only the path and optional query, not a fragment ([RFC 9112 section 3.2.1](https://www.rfc-editor.org/rfc/rfc9112.html#section-3.2.1)). Browsers and URL-based clients omit the fragment before sending the request, so UI hash routing is unaffected

A literal hash in path or query data must be encoded as `%23`, which the gateway forwards unchanged. Direct Python with h11 also accepts unescaped hashes as data, but that permissive raw-target behavior is outside this URL contract. Active socket tests cover both [fragment omission](tests/exchange.rs) and [encoded-hash preservation](tests/exchange.rs)

## Known differences from uvicorn with h11

These cases remain executable as ignored socket tests. They are not passing parity checks

| Input | Direct Python | Through Rust | Regression test |
|---|---|---|---|
| Header after a removed hop-by-hop field | Keeps the remaining header order | `HeaderMap::remove` can move a later header | [dropping_a_hop_by_hop_header_keeps_the_others_in_order](tests/exchange.rs) |
| Repeated header names interleaved with other names | Keeps the interleaving | Groups values by name, keeping their order within that name | [interleaved_repeated_headers_reach_python_interleaved](tests/exchange.rs) |
| Obsolete folded header, such as `x-a: b\r\n c` | Unfolds to `b c` | Rejects with 400 | [a_folded_header_reaches_python_unfolded](tests/exchange.rs) |
| Identical Content-Length list, such as `7, 7` | Accepts length 7 | Rejects with 400 | [a_content_length_list_of_one_value_reaches_python](tests/framing.rs) |
| HTTP/1.0 request with chunked transfer coding | Decodes the body | Rejects with 400 | [an_http10_chunked_upload_reaches_python](tests/framing.rs) |

Clients can avoid the parser rejections by sending unfolded headers, one Content-Length value, and HTTP/1.1 for chunked uploads. Integrations must not depend on header ordering across different field names

Run `cargo test -p litellm-gateway-forward --test exchange --test framing -- --ignored` from this directory to reproduce the differences

## Connection header handling

The gateway strips standard hop-by-hop fields and fields named by Connection, except routing, representation, credential, forwarding, and WebSocket handshake headers. A client cannot nominate those end-to-end fields for removal. The private `x-litellm-rust-peer` always replaces any client-supplied copy
