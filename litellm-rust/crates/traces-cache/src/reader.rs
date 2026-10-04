use std::{collections::BTreeMap, sync::Arc, time::Duration};

use litellm_traces::{
    ObservationType, QueryScope, SpanDetail, SpanErrorPage, Trace, TracePage, resolve_trace,
    search::{
        MAX_HISTOGRAM_BUCKETS, MAX_RUN_VALUES, RunField, RunFilter, RunValues, TraceHistogram,
        histogram,
    },
    store::{
        CountBy, CountValue, RunCountQuery, RunQuery, RunSelection, SpanPart, SpanQuery, SpanRow,
        SpanSelection, SpanText, SpanTextQuery,
    },
    to_ui_content,
};

use crate::{
    ReadError, Snapshot, SnapshotCache, SnapshotKey, StoreError, TraceStore,
    cache::{Freshness, ListCache},
    cursor::{Cursor, SpanPosition, TextPosition, run_position, span_position, text_position},
    list::{list_summaries, run_batches},
    pages::read_all,
    spend::spend,
};

pub const MAX_GRAPH_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_GRAPH_SPANS: usize = 100_000;

const SNAPSHOT_IDLE: Duration = Duration::from_secs(120);
const ERROR_PAGE_CHARS: u64 = 16_384;

#[derive(Clone, Debug, Default)]
pub struct PageRequest {
    pub cursor: Option<String>,
    pub limit: u32,
}

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
        access: &QueryScope,
        filter: &RunFilter,
        page: &PageRequest,
    ) -> Result<TracePage, ReadError<S::Error>> {
        if page.limit == 0 || filter.start_ms >= filter.end_ms {
            return Err(ReadError::InvalidParameters);
        }
        let after = run_position(page.cursor.as_deref())?;
        let scope = SnapshotKey::scope(store.source(), access)?;
        let accepted = self.lists.limits.get(&scope).await.unwrap_or(u32::MAX);
        let mut query = RunQuery {
            selection: RunSelection::Matching(filter.clone()),
            after,
            limit: page.limit.min(500).min(accepted),
        };
        let rows = loop {
            match store.runs(access, &query).await {
                Err(StoreError::TooLarge) if query.limit > 1 => {
                    query.limit /= 2;
                    self.lists.limits.insert(scope.clone(), query.limit).await;
                }
                Err(StoreError::TooLarge) => return Err(ReadError::TooLarge),
                result => break result.map_err(map_store_error)?,
            }
        };
        let next_cursor = rows
            .last()
            .filter(|_| rows.len() == query.limit as usize)
            .map(|last| Cursor::Run(last.cursor()).encode());
        let data = {
            let mut summaries = Vec::with_capacity(rows.len());
            for batch in run_batches(&rows) {
                summaries.extend(list_summaries(self, store, access, batch).await?);
            }
            summaries
        };
        Ok(TracePage { data, next_cursor })
    }

    pub async fn histogram<S: TraceStore>(
        &self,
        store: &S,
        access: &QueryScope,
        filter: &RunFilter,
        buckets: u32,
    ) -> Result<TraceHistogram, ReadError<S::Error>> {
        if filter.start_ms >= filter.end_ms || !(1..=MAX_HISTOGRAM_BUCKETS).contains(&buckets) {
            return Err(ReadError::InvalidParameters);
        }
        let query = RunCountQuery {
            filter: filter.clone(),
            by: CountBy {
                buckets: Some(buckets),
                failed: true,
                value: Some(CountValue::PrimaryAgent),
            },
            contains: String::new(),
            limit: None,
        };
        let rows = store
            .run_counts(access, &query)
            .await
            .map_err(map_store_error)?;
        Ok(histogram(&rows, filter.start_ms, filter.end_ms, buckets))
    }

    pub async fn values<S: TraceStore>(
        &self,
        store: &S,
        access: &QueryScope,
        filter: &RunFilter,
        field: RunField,
        contains: &str,
        limit: u32,
    ) -> Result<RunValues, ReadError<S::Error>> {
        if filter.start_ms >= filter.end_ms || !(1..=MAX_RUN_VALUES).contains(&limit) {
            return Err(ReadError::InvalidParameters);
        }
        let query = RunCountQuery {
            filter: filter.clone(),
            by: CountBy {
                value: Some(CountValue::Field(field)),
                ..CountBy::default()
            },
            contains: contains.to_owned(),
            limit: Some(limit),
        };
        let rows = store
            .run_counts(access, &query)
            .await
            .map_err(map_store_error)?;
        Ok(RunValues {
            values: rows.into_iter().map(|row| row.value).collect(),
        })
    }

    pub async fn get_trace<S: TraceStore>(
        &self,
        store: &S,
        access: &QueryScope,
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
        access: &QueryScope,
        trace_id: &str,
        trace_ref: &str,
        cursor: Option<&str>,
        page_size: u32,
    ) -> Result<Option<Trace>, ReadError<S::Error>> {
        if !(1..=500).contains(&page_size) {
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
        let position = span_position(cursor)?;
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
        access: &QueryScope,
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
        access: &QueryScope,
        trace_id: &str,
        trace_ref: &str,
        snapshot_ms: u64,
    ) -> Result<Arc<Snapshot>, Arc<Miss<S::Error>>> {
        let key = SnapshotKey::new(store.source(), access, trace_id, trace_ref, snapshot_ms)
            .map_err(|error| Arc::new(error.into()))?;
        self.snapshots
            .pinned_or_load(key, snapshot_ms, async {
                let selection = SpanSelection::Trace {
                    trace_id: trace_id.to_owned(),
                    trace_ref: trace_ref.to_owned(),
                };
                let rows = spans(store, access, selection, snapshot_ms)
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
        access: &QueryScope,
        trace_id: &str,
        span_id: &str,
        trace_ref: &str,
    ) -> Result<Option<SpanDetail>, ReadError<S::Error>> {
        let Some(trace_ref) = reference(store, access, trace_id, trace_ref).await? else {
            return Ok(None);
        };
        let read = |part| {
            let query = SpanTextQuery {
                trace_id: trace_id.to_owned(),
                trace_ref: trace_ref.clone(),
                span_id: span_id.to_owned(),
                part,
                offset: 0,
                max_chars: None,
            };
            async move { store.span_text(access, &query).await }
        };
        let Some(input) = read(SpanPart::Input).await.map_err(map_store_error)? else {
            return Ok(None);
        };
        let output = text_of(read(SpanPart::Output).await)?;
        let attributes = text_of(read(SpanPart::Attributes).await)?;
        let output = if output.is_empty() {
            self.agent_answer(store, access, trace_id, &trace_ref, span_id)
                .await?
        } else {
            output
        };
        Ok(Some(SpanDetail {
            input_ui: to_ui_content(&input.text),
            output_ui: to_ui_content(&output),
            span_id: span_id.to_owned(),
            input: input.text,
            output,
            attributes: parse_attributes(&attributes)?,
        }))
    }

    async fn agent_answer<S: TraceStore>(
        &self,
        store: &S,
        access: &QueryScope,
        trace_id: &str,
        trace_ref: &str,
        span_id: &str,
    ) -> Result<String, ReadError<S::Error>> {
        let Some(snapshot) = self.current(store, access, trace_id, trace_ref).await? else {
            return Ok(String::new());
        };
        let spans = &snapshot.trace().spans;
        if !spans
            .iter()
            .any(|span| span.span_id == span_id && span.kind == ObservationType::Agent)
        {
            return Ok(String::new());
        }
        let mut calls: Vec<_> = spans
            .iter()
            .filter(|span| {
                span.kind == ObservationType::Llm && span.parent_span_id.as_deref() == Some(span_id)
            })
            .collect();
        calls.sort_by(|left, right| right.start_offset_ms.total_cmp(&left.start_offset_ms));
        for call in calls {
            let query = SpanTextQuery {
                trace_id: trace_id.to_owned(),
                trace_ref: trace_ref.to_owned(),
                span_id: call.span_id.clone(),
                part: SpanPart::Output,
                offset: 0,
                max_chars: None,
            };
            let output = text_of(store.span_text(access, &query).await)?;
            if !output.is_empty() {
                return Ok(output);
            }
        }
        Ok(String::new())
    }

    pub async fn get_span_error<S: TraceStore>(
        &self,
        store: &S,
        access: &QueryScope,
        trace_id: &str,
        span_id: &str,
        trace_ref: &str,
        cursor: Option<&str>,
    ) -> Result<Option<SpanErrorPage>, ReadError<S::Error>> {
        let position = text_position(cursor, SpanPart::Error)?;
        let Some(trace_ref) = reference(store, access, trace_id, trace_ref).await? else {
            return Ok(None);
        };
        let offset = position.as_ref().map_or(0, |position| position.offset);
        let query = SpanTextQuery {
            trace_id: trace_id.to_owned(),
            trace_ref,
            span_id: span_id.to_owned(),
            part: SpanPart::Error,
            offset,
            max_chars: Some(ERROR_PAGE_CHARS),
        };
        let Some(text) = store
            .span_text(access, &query)
            .await
            .map_err(map_store_error)?
        else {
            return Ok(None);
        };
        if position.is_some_and(|position| position.version != text.version) {
            return Ok(None);
        }
        let next_offset = offset + text.text.chars().count() as u64;
        let next_cursor = (next_offset < text.total_chars).then(|| {
            Cursor::Text(TextPosition {
                part: SpanPart::Error,
                offset: next_offset,
                version: text.version,
            })
            .encode()
        });
        Ok(Some(SpanErrorPage {
            span_id: span_id.to_owned(),
            message: text.text,
            total_chars: text.total_chars,
            next_cursor,
        }))
    }
}

fn text_of<E>(result: Result<Option<SpanText>, StoreError<E>>) -> Result<String, ReadError<E>> {
    Ok(result
        .map_err(map_store_error)?
        .map(|text| text.text)
        .unwrap_or_default())
}

fn parse_attributes<E>(json: &str) -> Result<BTreeMap<String, String>, ReadError<E>> {
    if json.is_empty() {
        return Ok(BTreeMap::new());
    }
    serde_json::from_str(json).map_err(|error| ReadError::Encode(Arc::new(error)))
}

pub(super) async fn spans<S: TraceStore>(
    store: &S,
    access: &QueryScope,
    selection: SpanSelection,
    as_of_ms: u64,
) -> Result<Vec<SpanRow>, StoreError<S::Error>> {
    let mut rows = read_all(|after, limit| {
        let query = SpanQuery {
            selection: selection.clone(),
            as_of_ms,
            after,
            limit,
        };
        async move { store.spans(access, &query).await }
    })
    .await?;
    rows.sort_by_key(|row| row.start_ns);
    Ok(rows)
}

async fn reference<S: TraceStore>(
    store: &S,
    access: &QueryScope,
    trace_id: &str,
    trace_ref: &str,
) -> Result<Option<String>, ReadError<S::Error>> {
    if !trace_ref.is_empty() {
        return Ok(Some(trace_ref.to_owned()));
    }
    let query = RunQuery {
        selection: RunSelection::TraceId(trace_id.to_owned()),
        after: None,
        limit: 2,
    };
    let runs = store.runs(access, &query).await.map_err(map_store_error)?;
    if runs.len() > 1 {
        return Err(ReadError::AmbiguousTrace);
    }
    Ok(runs.into_iter().next().map(|run| run.trace_ref))
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
                Cursor::Span(SpanPosition {
                    trace_ref: position.trace_ref.clone(),
                    snapshot_ms: position.snapshot_ms,
                    offset: end,
                    version: snapshot.version().to_owned(),
                })
                .encode()
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
