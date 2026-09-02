pub mod generated {
    tonic::include_proto!("litellm.python_extension.v1");
}

pub use generated::*;

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::{CacheRef, GuardrailDecision, StreamFrame, StreamFrameKind};

    #[test]
    fn cache_reference_round_trips_without_gateway_objects() -> Result<(), prost::DecodeError> {
        let reference = CacheRef {
            invocation_id: "invocation-1".to_string(),
            opaque_handle: "opaque".to_string(),
        };
        assert_eq!(
            CacheRef::decode(reference.encode_to_vec().as_slice())?,
            reference
        );
        Ok(())
    }

    #[test]
    fn duplex_frame_round_trips() -> Result<(), prost::DecodeError> {
        let frame = StreamFrame {
            kind: StreamFrameKind::InputChunk.into(),
            stream_id: "stream-1".to_string(),
            chunk_json: Some(br#"{"value":"hello"}"#.to_vec()),
            ..Default::default()
        };
        assert_eq!(
            StreamFrame::decode(frame.encode_to_vec().as_slice())?,
            frame
        );
        Ok(())
    }

    #[test]
    fn block_and_transport_error_are_distinct_outcomes() {
        assert_ne!(
            GuardrailDecision::Block as i32,
            GuardrailDecision::Error as i32
        );
    }
}
