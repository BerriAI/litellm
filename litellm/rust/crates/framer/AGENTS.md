# framer

Splits a provider's byte stream into the units of its wire format. It knows SSE and the AWS event stream, and nothing about what a unit's payload means

- Names follow the SSE spec, and the module doc of `src/sse.rs` is the glossary. The Python proxy calls the raw text up to a blank line a "frame" (`_restamped_frame` and `_SSE_FRAME_END_PATTERN` in `litellm/proxy/anthropic_endpoints/streaming_model_restamp.py`). Do not carry that name over when porting from it. SSE has no frames, the spec's examples call that unit a block, and here frame only means what a tokio-util codec decodes
- No JSON in this crate. `SseEvent.data` is text, and a caller that needs it as JSON parses it on its own side, so no route's event format leaks in here
- A relay never rebuilds a block from its `SseEvent`. An earlier model restamp did, and it dropped the block's comments, its line endings and the order of its fields. To change a relayed event, use `SseBlock::with_data` or add an operation as narrow as it, never a constructor from `SseEvent`
- `blocks` neither adds nor drops a byte, and its tests pin that for each line terminator at several chunk sizes. Keep that true of anything new that hands out an `SseBlock`
