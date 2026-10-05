use std::{sync::Arc, time::Duration};

use crate::{
    ReadError, Snapshot, SnapshotCache, SnapshotKey, StoreError, TraceStore,
    cache::{Freshness, ListCache},
    cursor::{
        ErrorPosition, SpanPosition, decode_cursor, encode_cursor, error_position, trace_position,
    },
    list::{list_summaries, run_batches},
    spend::spend,
};
use litellm_traces::{
    SpanDetail, SpanErrorPage, Trace, TracePage,
    query::named::{
        ListTracesParams, ReadAccessParams, SpanDetailParams, SpanErrorParams, TraceIdentityParams,
        TraceSpansParams,
    },
    request::{TRACE_PAGE_SIZE_MAX, TRACE_PAGE_SIZE_MIN},
    resolve_trace, to_ui_content,
};

pub const MAX_GRAPH_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_GRAPH_SPANS: usize = 100_000;

const SNAPSHOT_IDLE: Duration = Duration::from_secs(120);

/// A read that found no trace, kept apart from failures so single-flight waiters share it
/// without it being cached.
pub(super) enum Miss<E> {
    Absent,
    Read(ReadError<E>),
}

impl<E> From<crate::Error> for Miss<E> {
    fn from(error: crate::Error) -> Self {
        Self::Read(error.into())
    }
}

fn settle<T, E>(result: Result<T, Arc<Miss<E>>>) -> Result<Option<T>, ReadError<E>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(miss) => match &*miss {
            Miss::Absent => Ok(None),
            Miss::Read(error) => Err(error.clone()),
        },
    }
}

pub struct TraceReader {
    snapshots: SnapshotCache,
    pub(super) lists: ListCache,
    response_bytes: usize,
}

impl TraceReader {
    pub fn new(response_bytes: usize) -> Self {
        Self {
            snapshots: SnapshotCache::new(MAX_GRAPH_BYTES, SNAPSHOT_IDLE),
            lists: ListCache::new(),
            response_bytes,
        }
    }

    pub async fn list_traces<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        start_ms: i64,
        end_ms: i64,
        cursor: Option<&str>,
        limit: u32,
    ) -> Result<TracePage, ReadError<S::Error>> {
        if limit == 0 {
            return Err(ReadError::InvalidParameters);
        }
        let (cursor_ms, cursor_trace_id) = trace_position(cursor)?;
        let scope = SnapshotKey::scope(store.source(), access)?;
        let accepted = self.lists.limits.get(&scope).await.unwrap_or(u32::MAX);
        let mut params = ListTracesParams {
            access: access.clone(),
            start_ms,
            end_ms,
            cursor_ms,
            cursor_trace_id,
            limit: limit.min(500).min(accepted),
        };
        let page = loop {
            match store.list_runs(&params).await {
                Err(StoreError::TooLarge) if params.limit > 1 => {
                    params.limit /= 2;
                    self.lists.limits.insert(scope.clone(), params.limit).await;
                }
                Err(StoreError::TooLarge) => return Err(ReadError::TooLarge),
                result => break result.map_err(map_store_error)?,
            }
        };
        let next_cursor = page
            .last()
            .filter(|_| page.len() == params.limit as usize)
            .map(|last| encode_cursor(&(last.start_ms, &last.trace_ref)));
        let data = {
            let mut summaries = Vec::with_capacity(page.len());
            for batch in run_batches(&page) {
                summaries.extend(list_summaries(self, store, access, batch).await?);
            }
            summaries
        };
        Ok(TracePage { data, next_cursor })
    }

    pub async fn get_trace<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
    ) -> Result<Option<Trace>, ReadError<S::Error>> {
        let Some(trace_ref) = reference(store, access, trace_id, trace_ref).await? else {
            return Ok(None);
        };
        Ok(self
            .current(store, access, trace_id, &trace_ref)
            .await?
            .map(|snapshot| snapshot.trace().clone()))
    }

    pub async fn get_trace_page<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
        cursor: Option<&str>,
        page_size: u32,
    ) -> Result<Option<Trace>, ReadError<S::Error>> {
        if !(u32::from(TRACE_PAGE_SIZE_MIN)..=u32::from(TRACE_PAGE_SIZE_MAX)).contains(&page_size) {
            return Err(ReadError::InvalidParameters);
        }
        let Some(trace_ref) = reference(store, access, trace_id, trace_ref).await? else {
            return Ok(None);
        };
        let Some(cursor) = cursor else {
            let Some(snapshot) = self.current(store, access, trace_id, &trace_ref).await? else {
                return Ok(None);
            };
            let position = SpanPosition {
                trace_ref,
                snapshot_ms: snapshot.snapshot_ms(),
                offset: 0,
                version: snapshot.version().to_owned(),
            };
            return page(&snapshot, &position, page_size, self.response_bytes).map(Some);
        };
        let position: SpanPosition = decode_cursor(cursor, "span")?;
        if position.trace_ref != trace_ref || position.snapshot_ms == 0 {
            return Err(ReadError::InvalidCursor("span"));
        }
        let Some(snapshot) = settle(
            self.pinned(store, access, trace_id, &trace_ref, position.snapshot_ms)
                .await,
        )?
        else {
            return Ok(None);
        };
        if position.version != snapshot.version() {
            return Err(ReadError::TraceChanged);
        }
        if position.offset > snapshot.trace().spans.len() {
            return Err(ReadError::InvalidCursor("span"));
        }
        page(&snapshot, &position, page_size, self.response_bytes).map(Some)
    }

    pub(super) async fn current<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
    ) -> Result<Option<Arc<Snapshot>>, ReadError<S::Error>> {
        let latest = SnapshotKey::latest(store.source(), access, trace_id, trace_ref)?;
        settle(
            self.snapshots
                .latest_or_load(latest, now_ms(), |snapshot_ms| {
                    self.pinned(store, access, trace_id, trace_ref, snapshot_ms)
                })
                .await,
        )
    }

    async fn pinned<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
        snapshot_ms: u64,
    ) -> Result<Arc<Snapshot>, Arc<Miss<S::Error>>> {
        let key = SnapshotKey::new(store.source(), access, trace_id, trace_ref, snapshot_ms)
            .map_err(|error| Arc::new(error.into()))?;
        self.snapshots
            .pinned_or_load(key, snapshot_ms, async {
                let params = TraceSpansParams {
                    access: access.clone(),
                    trace_id: trace_id.to_owned(),
                    trace_ref: trace_ref.to_owned(),
                };
                let rows = store
                    .trace_spans(&params, snapshot_ms)
                    .await
                    .map_err(|error| Miss::Read(map_store_error(error)))?;
                let spend_rows = spend(store, access, &rows).await;
                let freshness = Freshness::of(&rows, spend_rows.is_some(), snapshot_ms);
                resolve_trace(
                    trace_id,
                    trace_ref,
                    &rows,
                    spend_rows.as_deref().unwrap_or_default(),
                )
                .map(|trace| (trace, freshness))
                .ok_or(Miss::Absent)
            })
            .await
    }

    pub async fn get_span<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        trace_id: &str,
        span_id: &str,
        trace_ref: &str,
    ) -> Result<Option<SpanDetail>, ReadError<S::Error>> {
        let Some(trace_ref) = reference(store, access, trace_id, trace_ref).await? else {
            return Ok(None);
        };
        let params = SpanDetailParams {
            access: access.clone(),
            trace_id: trace_id.to_owned(),
            trace_ref,
            span_id: span_id.to_owned(),
        };
        let row = store.span_detail(&params).await.map_err(map_store_error)?;
        Ok(row.map(|row| SpanDetail {
            input_ui: to_ui_content(&row.input),
            output_ui: to_ui_content(&row.output),
            span_id: row.span_id,
            input: row.input,
            output: row.output,
            attributes: row.attributes,
        }))
    }

    pub async fn get_span_error<S: TraceStore>(
        &self,
        store: &S,
        access: &ReadAccessParams,
        trace_id: &str,
        span_id: &str,
        trace_ref: &str,
        cursor: Option<&str>,
    ) -> Result<Option<SpanErrorPage>, ReadError<S::Error>> {
        let position = error_position(cursor)?;
        let Some(trace_ref) = reference(store, access, trace_id, trace_ref).await? else {
            return Ok(None);
        };
        let offset = position.as_ref().map_or(0, |position| position.offset);
        let params = SpanErrorParams {
            access: access.clone(),
            trace_id: trace_id.to_owned(),
            trace_ref,
            span_id: span_id.to_owned(),
            error_offset: offset,
            error_version: position
                .map(|position| position.version)
                .unwrap_or_default(),
        };
        let Some(row) = store.span_error(&params).await.map_err(map_store_error)? else {
            return Ok(None);
        };
        let next_offset = offset + row.message.chars().count() as u64;
        let next_cursor = (next_offset < row.total_chars).then(|| {
            encode_cursor(&ErrorPosition {
                offset: next_offset,
                version: row.version,
            })
        });
        Ok(Some(SpanErrorPage {
            span_id: row.span_id,
            message: row.message,
            total_chars: row.total_chars,
            next_cursor,
        }))
    }
}

async fn reference<S: TraceStore>(
    store: &S,
    access: &ReadAccessParams,
    trace_id: &str,
    trace_ref: &str,
) -> Result<Option<String>, ReadError<S::Error>> {
    if !trace_ref.is_empty() {
        return Ok(Some(trace_ref.to_owned()));
    }
    let params = TraceIdentityParams {
        access: access.clone(),
        trace_id: trace_id.to_owned(),
    };
    let identities = store.trace_refs(&params).await.map_err(map_store_error)?;
    if identities.len() > 1 {
        return Err(ReadError::AmbiguousTrace);
    }
    Ok(identities.into_iter().next())
}

fn page<E>(
    snapshot: &Snapshot,
    position: &SpanPosition,
    page_size: u32,
    response_bytes: usize,
) -> Result<Trace, ReadError<E>> {
    let spans = &snapshot.trace().spans;
    let create_page = |count: usize| {
        let end = position.offset.saturating_add(count).min(spans.len());
        Trace {
            summary: snapshot.trace().summary.clone(),
            agents: snapshot.trace().agents.clone(),
            spans: spans[position.offset..end].to_vec(),
            next_cursor: (end < spans.len()).then(|| {
                encode_cursor(&SpanPosition {
                    trace_ref: position.trace_ref.clone(),
                    snapshot_ms: position.snapshot_ms,
                    offset: end,
                    version: snapshot.version().to_owned(),
                })
            }),
        }
    };
    let mut trace = create_page(page_size as usize);
    loop {
        if serde_json::to_vec(&trace)
            .map_err(|error| ReadError::Encode(Arc::new(error)))?
            .len()
            <= response_bytes
        {
            return Ok(trace);
        }
        if trace.spans.len() <= 1 {
            return Err(ReadError::TooLarge);
        }
        trace = create_page(trace.spans.len() / 2);
    }
}

pub(super) fn now_ms() -> u64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as u64
}

pub(super) fn map_store_error<E>(error: StoreError<E>) -> ReadError<E> {
    match error {
        StoreError::TooLarge => ReadError::TooLarge,
        StoreError::Failed(error) => ReadError::Store(Arc::new(error)),
    }
}
