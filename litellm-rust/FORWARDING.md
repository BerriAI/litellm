# Python forwarding compatibility

The opt-in forwarding gateway serves HTTP/1.1 and WebSockets in front of Python. Python still owns every route and policy. Its protocol contract and environment settings are in [gateway-forward/AGENTS.md](crates/gateway-forward/AGENTS.md)

## Known differences from uvicorn with h11

These cases remain executable as ignored socket tests. They are not passing parity checks

| Input | Direct Python | Through Rust | Regression test |
|---|---|---|---|
| Literal `#` in a request path or query | Keeps every byte | `http::Uri` truncates at `#`, potentially changing the route or query. Must be fixed before shipping | [forwards_an_unescaped_hash_in_the_target](crates/gateway-forward/tests/exchange.rs) |
| Header after a removed hop-by-hop field | Keeps the remaining header order | `HeaderMap::remove` can move a later header | [dropping_a_hop_by_hop_header_keeps_the_others_in_order](crates/gateway-forward/tests/exchange.rs) |
| Repeated header names interleaved with other names | Keeps the interleaving | Groups values by name, keeping their order within that name | [interleaved_repeated_headers_reach_python_interleaved](crates/gateway-forward/tests/exchange.rs) |
| Obsolete folded header, such as `x-a: b\r\n c` | Unfolds to `b c` | Rejects with 400 | [a_folded_header_reaches_python_unfolded](crates/gateway-forward/tests/exchange.rs) |
| Identical Content-Length list, such as `7, 7` | Accepts length 7 | Rejects with 400 | [a_content_length_list_of_one_value_reaches_python](crates/gateway-forward/tests/framing.rs) |
| HTTP/1.0 request with chunked transfer coding | Decodes the body | Rejects with 400 | [an_http10_chunked_upload_reaches_python](crates/gateway-forward/tests/framing.rs) |

Clients can avoid the parser rejections by sending unfolded headers, one Content-Length value, and HTTP/1.1 for chunked uploads. Integrations must not depend on header ordering across different field names

Run `cargo test -p litellm-gateway-forward --test exchange --test framing -- --ignored` from this directory to reproduce the differences

## Connection header handling

The gateway strips standard hop-by-hop fields and fields named by Connection, except routing, representation, credential, forwarding, and WebSocket handshake headers. A client cannot nominate those end-to-end fields for removal. The private `x-litellm-rust-peer` always replaces any client-supplied copy
