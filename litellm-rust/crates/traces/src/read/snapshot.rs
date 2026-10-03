use std::{sync::Arc, time::Duration};

use moka::future::Cache;
use sha2::{Digest, Sha256};

use crate::{ReadError, Trace};

use super::{MAX_GRAPH_BYTES, cursor::SpanPosition};

pub(super) struct Snapshot {
    pub(super) trace: Trace,
    pub(super) version: String,
    weight: u32,
}

impl Snapshot {
    pub(super) fn new<E>(trace: Trace) -> Result<Self, ReadError<E>> {
        let encoded = serde_json::to_vec(&trace).map_err(ReadError::Encode)?;
        if encoded.len() > MAX_GRAPH_BYTES {
            return Err(ReadError::TooLarge);
        }
        let span_ids: Vec<&str> = trace
            .spans
            .iter()
            .map(|span| span.span_id.as_str())
            .collect();
        let version = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&span_ids).map_err(ReadError::Encode)?)
        );
        Ok(Self {
            trace,
            version,
            weight: u32::try_from(encoded.len() * 2).unwrap_or(u32::MAX),
        })
    }
}

pub(super) fn cache() -> Cache<String, Arc<Snapshot>> {
    Cache::builder()
        .max_capacity((2 * MAX_GRAPH_BYTES) as u64)
        .weigher(|_: &String, snapshot: &Arc<Snapshot>| snapshot.weight)
        .time_to_live(Duration::from_secs(120))
        .build()
}

pub(super) fn page<E>(
    snapshot: &Snapshot,
    position: &SpanPosition,
    page_size: u32,
    response_bytes: usize,
) -> Result<Trace, ReadError<E>> {
    let spans = &snapshot.trace.spans;
    let create_page = |count: usize| {
        let end = position.offset.saturating_add(count).min(spans.len());
        Trace {
            summary: snapshot.trace.summary.clone(),
            agents: snapshot.trace.agents.clone(),
            spans: spans[position.offset..end].to_vec(),
            next_cursor: (end < spans.len()).then(|| {
                super::cursor::encode_cursor(&SpanPosition {
                    trace_ref: position.trace_ref.clone(),
                    snapshot_ms: position.snapshot_ms,
                    offset: end,
                    version: snapshot.version.clone(),
                })
            }),
        }
    };
    let mut trace = create_page(page_size as usize);
    loop {
        if serde_json::to_vec(&trace).map_err(ReadError::Encode)?.len() <= response_bytes {
            return Ok(trace);
        }
        if trace.spans.len() <= 1 {
            return Err(ReadError::TooLarge);
        }
        trace = create_page(trace.spans.len() / 2);
    }
}
