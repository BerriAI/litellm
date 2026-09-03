//! Header and upstream-body helpers shared by every route module.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::stream::Stream;
use serde_json::{Map, Value};

use crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS;
use crate::error::{Error, json_type_name};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn http_request(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, reqwest::Error> {
    request.send().await
}

/// A stream of complete server-sent-event frames.
///
/// Wraps an upstream byte stream and reassembles it into frames delimited by a
/// blank line (`\n\n` or `\r\n\r\n`), so a host receives whole `event:`/`data:`
/// frames regardless of how the provider chunked the response. Frames that are
/// only whitespace (keep-alive newlines) are dropped. A trailing partial frame
/// at end-of-stream is emitted as-is rather than lost.
pub struct SseFrameStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    buffer: Vec<u8>,
    done: bool,
}

impl SseFrameStream {
    pub fn new(response: reqwest::Response) -> Self {
        Self::from_byte_stream(Box::pin(response.bytes_stream()))
    }

    fn from_byte_stream(inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
            done: false,
        }
    }
}

impl Stream for SseFrameStream {
    type Item = Result<Vec<u8>, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(end) = find_frame_end(&self.buffer) {
                let frame: Vec<u8> = self.buffer.drain(..end).collect();
                if !frame.iter().all(u8::is_ascii_whitespace) {
                    return Poll::Ready(Some(Ok(frame)));
                }
                continue;
            }
            if self.done {
                if self.buffer.is_empty() {
                    return Poll::Ready(None);
                }
                let frame = std::mem::take(&mut self.buffer);
                return Poll::Ready(Some(Ok(frame)));
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => self.buffer.extend_from_slice(&chunk),
                Poll::Ready(Some(Err(error))) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(Error::Network(error.to_string()))));
                }
                Poll::Ready(None) => self.done = true,
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn find_frame_end(buffer: &[u8]) -> Option<usize> {
    for index in 0..buffer.len() {
        if buffer[index..].starts_with(b"\r\n\r\n") {
            return Some(index + 4);
        }
        if buffer[index..].starts_with(b"\n\n") {
            return Some(index + 2);
        }
    }
    None
}

/// Bound an upstream error body before it crosses a host boundary, so provider
/// bodies stay data-minimized.
pub fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= UPSTREAM_ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(UPSTREAM_ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub fn string_headers(
    context: &'static str,
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    Error::InvalidRequest(format!(
                        "{context} extra_headers.{key} must be a string, got {}",
                        json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

pub fn has_bearer_auth(headers: &[(String, String)]) -> bool {
    headers.iter().any(|(name, value)| {
        if !name.eq_ignore_ascii_case("authorization") {
            return false;
        }
        let value = value.trim();
        value.len() > 7
            && value[..7].eq_ignore_ascii_case("bearer ")
            && !value[7..].trim().is_empty()
    })
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures_util::StreamExt;
    use serde_json::json;

    use super::*;

    fn frame_stream(chunks: Vec<&[u8]>) -> SseFrameStream {
        let chunks: Vec<reqwest::Result<Bytes>> = chunks
            .into_iter()
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect();
        SseFrameStream::from_byte_stream(Box::pin(futures_util::stream::iter(chunks)))
    }

    async fn collect_frames(chunks: Vec<&[u8]>) -> Vec<Vec<u8>> {
        let mut stream = frame_stream(chunks);
        let mut frames = Vec::new();
        while let Some(frame) = stream.next().await {
            frames.push(frame.expect("frame should decode"));
        }
        frames
    }

    #[test]
    fn truncate_leaves_short_bodies_untouched() {
        assert_eq!(truncate_error_body("short"), "short");
    }

    #[test]
    fn truncate_bounds_long_bodies_by_characters() {
        let body = "\u{00e9}".repeat(UPSTREAM_ERROR_BODY_MAX_CHARS + 10);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("... (truncated)"));
        assert_eq!(
            truncated.chars().count(),
            UPSTREAM_ERROR_BODY_MAX_CHARS + "... (truncated)".chars().count()
        );
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = Map::from_iter([("x-trace".to_string(), json!(7))]);
        let err = string_headers("chat completions", Some(headers)).expect_err("non-string value");
        assert_eq!(
            err,
            Error::InvalidRequest(
                "chat completions extra_headers.x-trace must be a string, got number".to_string()
            )
        );
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let headers = vec![("X-Api-Key".to_string(), "k".to_string())];
        assert!(has_header(&headers, "x-api-key"));
        assert!(!has_header(&headers, "authorization"));
    }

    #[test]
    fn bearer_detection_requires_a_non_empty_token() {
        assert!(has_bearer_auth(&[(
            "Authorization".to_string(),
            "Bearer abc".to_string()
        )]));
        assert!(!has_bearer_auth(&[(
            "Authorization".to_string(),
            "Bearer    ".to_string()
        )]));
        assert!(!has_bearer_auth(&[(
            "Authorization".to_string(),
            "Basic abc".to_string()
        )]));
    }

    #[tokio::test]
    async fn sse_frames_split_on_blank_lines() {
        let frames = collect_frames(vec![b"event: a\ndata: {}\n\nevent: b\ndata: {}\n\n"]).await;
        assert_eq!(
            frames,
            vec![
                b"event: a\ndata: {}\n\n".to_vec(),
                b"event: b\ndata: {}\n\n".to_vec()
            ]
        );
    }

    #[tokio::test]
    async fn sse_frames_reassemble_across_chunk_boundaries() {
        let frames = collect_frames(vec![
            b"event: message_star",
            b"t\ndata: {\"a\":1}\n",
            b"\nevent: message_stop\ndata: {}\n",
            b"\n",
        ])
        .await;
        assert_eq!(
            frames,
            vec![
                b"event: message_start\ndata: {\"a\":1}\n\n".to_vec(),
                b"event: message_stop\ndata: {}\n\n".to_vec()
            ]
        );
    }

    #[tokio::test]
    async fn sse_frames_accept_crlf_delimiters() {
        let frames = collect_frames(vec![b"event: a\r\ndata: {}\r\n\r\n"]).await;
        assert_eq!(frames, vec![b"event: a\r\ndata: {}\r\n\r\n".to_vec()]);
    }

    #[tokio::test]
    async fn sse_frames_drop_whitespace_only_keep_alive_chunks() {
        let frames = collect_frames(vec![b"\n\nevent: a\ndata: {}\n\n\n\n"]).await;
        assert_eq!(frames, vec![b"event: a\ndata: {}\n\n".to_vec()]);
    }

    #[tokio::test]
    async fn sse_frames_emit_trailing_partial_frame_at_eof() {
        let frames = collect_frames(vec![b"event: a\ndata: {}\n\nevent: b\ndata: {"]).await;
        assert_eq!(
            frames,
            vec![
                b"event: a\ndata: {}\n\n".to_vec(),
                b"event: b\ndata: {".to_vec()
            ]
        );
    }

    #[tokio::test]
    async fn empty_input_yields_no_frames() {
        assert!(collect_frames(vec![]).await.is_empty());
    }
}
