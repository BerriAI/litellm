# framer

Splits a provider's byte stream into the units of its wire format. It knows SSE and the AWS event stream, and nothing about what a unit's payload means

- Names follow the SSE spec, and the module doc of `src/sse.rs` is the glossary. The Python proxy calls the raw text up to a blank line a "frame" (`_restamped_frame` and `_SSE_FRAME_END_PATTERN` in `litellm/proxy/anthropic_endpoints/streaming_model_restamp.py`). Do not carry that name over when porting from it. SSE has no frames, the spec's examples call that unit a block, and here frame only means what a tokio-util codec decodes
- No JSON in this crate. `SseEvent.data` is text, and a caller that needs it as JSON parses it on its own side, so no route's event format leaks in here
- `blocks` neither adds nor drops a byte, and hands out bytes and failed chunks in the order it read them. The property tests in `tests/sse.rs` pin that for any cut of the stream, along with two more rules: where the stream is cut never changes its blocks, and the events they carry are the ones `SseCodec` decodes. Keep all of that true of anything new that hands out an `SseBlock`
