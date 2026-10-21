use crate::{Error, control::JobClient, wire};
use futures_util::{Stream, TryStreamExt, stream};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{Arc, Mutex},
};
use tokio::io::AsyncWriteExt;
use unicode_casefold::UnicodeCaseFold;

pub const MAX_TOOL_BYTES: usize = 8 * 1024 * 1024;
const MAX_PYTHON_INPUT: usize = 256 * 1024 * 1024;

#[derive(Clone)]
pub struct Workspace {
    pub executions: Vec<wire::Execution>,
    pub reviews: Vec<wire::ReviewRecord>,
    pub client: JobClient,
    partial: Arc<Mutex<BTreeSet<String>>>,
    errors: Arc<Mutex<BTreeMap<String, BTreeSet<String>>>>,
    previews: Arc<Mutex<BTreeMap<String, Vec<wire::ReviewSpan>>>>,
}

struct Source {
    execution: wire::Execution,
    cursor: String,
    part: wire::TracePart,
}

impl Workspace {
    pub fn new(executions: Vec<wire::Execution>, client: JobClient) -> Self {
        Self {
            executions,
            client,
            reviews: Vec::new(),
            partial: Arc::default(),
            errors: Arc::default(),
            previews: Arc::default(),
        }
    }

    pub fn partial(&self, execution: &wire::Execution) -> bool {
        !execution.root_seen
            || self
                .partial
                .lock()
                .map(|p| p.contains(&execution.id))
                .unwrap_or(true)
    }

    pub fn errors(&self) -> Vec<String> {
        self.errors
            .lock()
            .map(|errors| {
                errors
                    .iter()
                    .flat_map(|(execution_id, errors)| {
                        errors
                            .iter()
                            .map(move |error| format!("{error} (execution {execution_id})"))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn read_failed(&self, execution_id: &str) -> bool {
        self.errors
            .lock()
            .map(|errors| errors.contains_key(execution_id))
            .unwrap_or(true)
    }

    pub fn previews(&self, execution_id: &str) -> Vec<wire::ReviewSpan> {
        self.previews
            .lock()
            .ok()
            .and_then(|previews| previews.get(execution_id).cloned())
            .unwrap_or_default()
    }

    fn incomplete(&self, execution: &wire::Execution, error: Error) -> Error {
        if let Ok(mut partial) = self.partial.lock() {
            partial.insert(execution.id.clone());
        }
        if let Ok(mut errors) = self.errors.lock() {
            errors
                .entry(execution.id.clone())
                .or_default()
                .insert(error.to_string());
        }
        error
    }

    async fn page(
        &self,
        execution: &wire::Execution,
        cursor: &str,
        offset: usize,
    ) -> Result<wire::ExecutionContent, Error> {
        let page = self
            .client
            .content(&execution.id, cursor, offset)
            .await
            .map_err(|_| self.incomplete(execution, Error::EvidenceUnavailable))?;
        if page.execution.id != execution.id
            || page.parts.iter().any(|p| p.execution_id != execution.id)
        {
            return Err(self.incomplete(execution, Error::EvidenceExecutionChanged));
        }
        if page.partial
            && !page.parts.iter().any(|p| p.truncated)
            && let Ok(mut partial) = self.partial.lock()
        {
            partial.insert(execution.id.clone());
        }
        Ok(page)
    }

    fn sources<'a>(
        &'a self,
        execution: &'a wire::Execution,
        spans: &'a [String],
    ) -> impl Stream<Item = Result<Source, Error>> + 'a {
        struct Cursor {
            cursor: String,
            next: Option<String>,
            seen: BTreeSet<String>,
            parts: VecDeque<wire::TracePart>,
            loaded: bool,
        }
        stream::try_unfold(
            Cursor {
                cursor: String::new(),
                next: None,
                seen: BTreeSet::new(),
                parts: VecDeque::new(),
                loaded: false,
            },
            move |mut state| async move {
                loop {
                    if let Some(part) = state.parts.pop_front() {
                        if spans.is_empty() || spans.contains(&part.span_id) {
                            return Ok(Some((
                                Source {
                                    execution: execution.clone(),
                                    cursor: state.cursor.clone(),
                                    part,
                                },
                                state,
                            )));
                        }
                        continue;
                    }
                    if state.loaded {
                        let Some(next) = state.next.take() else {
                            return Ok(None);
                        };
                        state.cursor = next;
                    }
                    if !state.seen.insert(state.cursor.clone()) {
                        return Err(self.incomplete(execution, Error::EvidenceCursorRepeated));
                    }
                    let page = self.page(execution, &state.cursor, 1).await?;
                    state.parts = page.parts.into();
                    state.next = page.next_cursor;
                    state.loaded = true;
                }
            },
        )
    }

    fn chunks<'a>(
        &'a self,
        source: &'a Source,
        start: usize,
    ) -> impl Stream<Item = Result<wire::TracePart, Error>> + 'a {
        stream::try_unfold(
            (true, true, start),
            move |(first, pending, offset)| async move {
                if !pending {
                    return Ok(None);
                }
                let part = if first && start == 0 {
                    source.part.clone()
                } else {
                    self.page(&source.execution, &source.cursor, offset + 1)
                        .await?
                        .parts
                        .into_iter()
                        .find(|p| p.span_id == source.part.span_id)
                        .ok_or_else(|| {
                            self.incomplete(&source.execution, Error::EvidenceSpanMissing)
                        })?
                };
                let characters = part.content.chars().count();
                if (!first && characters == 0) || (part.truncated && characters != 8000) {
                    return Err(self.incomplete(&source.execution, Error::EvidenceIncomplete));
                }
                let pending = part.truncated;
                Ok(Some((part, (false, pending, offset + 8000))))
            },
        )
    }

    async fn contains(&self, source: &Source, needle: &str, literal: bool) -> Result<bool, Error> {
        if needle.is_empty() {
            return Ok(!literal);
        }
        let needle = if literal {
            needle.to_owned()
        } else {
            needle.case_fold().collect()
        };
        let marker = "\n[... content omitted ...]\n";
        let delay = if literal { marker.len() - 1 } else { 0 };
        let mut tail = String::new();
        let chunks = self.chunks(source, 0);
        futures_util::pin_mut!(chunks);
        while let Some(piece) = chunks.try_next().await? {
            let text = tail
                + &if literal {
                    piece.content
                } else {
                    piece.content.case_fold().collect()
                };
            let segments: Vec<&str> = if literal {
                text.split(marker).collect()
            } else {
                vec![&text]
            };
            if segments[..segments.len() - 1]
                .iter()
                .any(|s| s.contains(&needle))
            {
                return Ok(true);
            }
            let last = segments[segments.len() - 1];
            let count = last.chars().count();
            if character_range(last, 0, Some(count.saturating_sub(delay))).contains(&needle) {
                return Ok(true);
            }
            tail = character_range(
                last,
                count.saturating_sub(needle.chars().count() - 1 + delay),
                None,
            );
        }
        Ok(tail.contains(&needle))
    }

    async fn ranged(
        &self,
        source: &Source,
        start: usize,
        end: Option<usize>,
        remaining: usize,
    ) -> Result<wire::TracePart, Error> {
        let mut content = String::new();
        let mut offset = start;
        let mut truncated = start > 0;
        let chunks = self.chunks(source, start);
        futures_util::pin_mut!(chunks);
        while let Some(piece) = chunks.try_next().await? {
            let size = piece.content.chars().count();
            let fragment =
                character_range(&piece.content, 0, end.map(|end| end.saturating_sub(offset)));
            if content.len().saturating_add(fragment.len()) > remaining {
                return Err(Error::ToolOutputTooLarge);
            }
            content.push_str(&fragment);
            offset += size;
            if end.is_some_and(|end| offset >= end) {
                truncated |= end.is_some_and(|end| offset > end) || piece.truncated;
                break;
            }
        }
        Ok(wire::TracePart {
            content,
            truncated,
            ..source.part.clone()
        })
    }

    pub async fn valid(&self, evidence: &wire::Evidence) -> Result<bool, Error> {
        let Some(execution) = self
            .executions
            .iter()
            .find(|e| e.id == evidence.execution_id)
        else {
            return Ok(false);
        };
        let selected = [evidence.span_id.clone()];
        let sources = self.sources(execution, &selected);
        futures_util::pin_mut!(sources);
        while let Some(source) = sources.try_next().await? {
            if self.contains(&source, &evidence.quote, true).await? {
                if let Ok(mut previews) = self.previews.lock() {
                    let entries = previews.entry(execution.id.clone()).or_default();
                    if entries.len() < 8
                        && !entries.iter().any(|p| p.span_id == source.part.span_id)
                    {
                        entries.push(serde_json::from_value(json!({"span_id": source.part.span_id, "name": character_range(&source.part.name, 0, Some(120)), "kind": character_range(&source.part.kind, 0, Some(40)), "preview": character_range(&evidence.quote, 0, Some(240)), "cited": true}))?);
                    }
                }
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub async fn fingerprint(&self, execution: &wire::Execution) -> Result<String, Error> {
        let mut digest = Sha256::new();
        digest.update(b"lens-rust-v1\0");
        digest.update(serde_json::to_vec(execution)?);
        let sources = self.sources(execution, &[]);
        futures_util::pin_mut!(sources);
        while let Some(source) = sources.try_next().await? {
            digest.update(serde_json::to_vec(&wire::TracePart {
                content: String::new(),
                truncated: false,
                ..source.part.clone()
            })?);
            let mut content_hash = Sha256::new();
            let chunks = self.chunks(&source, 0);
            futures_util::pin_mut!(chunks);
            while let Some(chunk) = chunks.try_next().await? {
                content_hash.update(chunk.content.as_bytes());
            }
            digest.update(content_hash.finalize());
        }
        digest.update([u8::from(self.partial(execution))]);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub async fn respond(&self, request: &wire::EvidenceRequest) -> Result<Value, Error> {
        use wire::EvidenceRequestAction as A;
        if request.char_end.is_some_and(|end| end < request.char_start) {
            return Ok(
                json!({"request": request, "error": "char_end must be at least char_start"}),
            );
        }
        if matches!(
            request.action,
            A::ReadReviews | A::ReviewCatalog | A::SearchReviews
        ) {
            return self.review_reply(request);
        }
        if request.action == A::Search && request.query.is_empty() {
            return Ok(
                json!({"request": request, "error": "Search requires nonempty literal text"}),
            );
        }
        let executions: Vec<_> = self
            .executions
            .iter()
            .filter(|e| request.execution_id.as_ref().is_none_or(|id| id == &e.id))
            .collect();
        if request.execution_id.is_some() && executions.is_empty() {
            return Ok(
                json!({"request": request, "error": "Unknown execution_id. Use the supplied catalog"}),
            );
        }
        let mut catalog = Vec::new();
        let mut parts = Vec::new();
        let mut missing: BTreeSet<_> = request.span_ids.iter().cloned().collect();
        let mut remaining = MAX_TOOL_BYTES;
        for execution in executions {
            if request.action == A::Catalog && request.execution_id.is_none() {
                catalog.push(json!({"execution": execution, "spans": [], "partial": self.partial(execution), "characters": null}));
                continue;
            }
            let sources = self.sources(execution, &request.span_ids);
            futures_util::pin_mut!(sources);
            let mut spans = Vec::new();
            while let Some(source) = sources.try_next().await? {
                missing.remove(&source.part.span_id);
                if request.action == A::Catalog {
                    let span = json!([
                        source.part.span_id,
                        source.part.parent_span_id,
                        source.part.name,
                        source.part.kind,
                        if source.part.truncated {
                            None
                        } else {
                            Some(source.part.content.chars().count())
                        },
                        source.part.start_time,
                        source.part.end_time
                    ]);
                    remaining = remaining
                        .checked_sub(serde_json::to_vec(&span)?.len())
                        .ok_or(Error::TooLarge)?;
                    spans.push(span);
                    continue;
                }
                if request.action == A::Search
                    && !self.contains(&source, &request.query, false).await?
                {
                    continue;
                }
                let part = self
                    .ranged(
                        &source,
                        request.char_start as usize,
                        request.char_end.map(|n| n as usize),
                        remaining,
                    )
                    .await?;
                remaining = remaining
                    .checked_sub(serde_json::to_vec(&part)?.len())
                    .ok_or(Error::TooLarge)?;
                parts.push(part);
            }
            if request.action == A::Catalog {
                catalog.push(json!({"execution": execution, "spans": spans, "partial": self.partial(execution), "characters": null}));
            }
        }
        let reply = json!({"request": request, "catalog": catalog, "parts": parts, "error": if missing.is_empty() || request.action == A::Catalog { String::new() } else { format!("Unknown span IDs: {}", missing.into_iter().collect::<Vec<_>>().join(", ")) }});
        limited(reply)
    }

    fn review_reply(&self, request: &wire::EvidenceRequest) -> Result<Value, Error> {
        use wire::EvidenceRequestAction as A;
        if request.action == A::SearchReviews && request.query.is_empty() {
            return Ok(
                json!({"request": request, "error": "Review search requires nonempty literal text"}),
            );
        }
        let selected: Vec<_> = self
            .reviews
            .iter()
            .filter(|r| {
                request
                    .execution_id
                    .as_ref()
                    .is_none_or(|id| id == &r.execution_id)
                    && request
                        .review_phase
                        .is_none_or(|p| p.to_string() == r.phase.to_string())
            })
            .collect();
        if request.action == A::ReviewCatalog {
            return limited(
                json!({"request": request, "review_catalog": selected.iter().map(|r| json!({"execution_id": r.execution_id, "phase": r.phase, "characters": r.content.chars().count()})).collect::<Vec<_>>() }),
            );
        }
        let needle: String = request.query.case_fold().collect();
        limited(
            json!({"request": request, "reviews": selected.into_iter().filter(|r| request.action != A::SearchReviews || r.content.case_fold().collect::<String>().contains(&needle)).map(|r| json!({"execution_id": r.execution_id, "phase": r.phase, "content": character_range(&r.content, request.char_start as usize, request.char_end.map(|n| n as usize))})).collect::<Vec<_>>() }),
        )
    }

    pub async fn python_input(
        &self,
        request: &wire::PythonRequest,
        file: &mut tokio::fs::File,
    ) -> Result<(), Error> {
        if request
            .execution_ids
            .iter()
            .any(|id| !self.executions.iter().any(|e| &e.id == id))
        {
            return Err(Error::UnknownPythonExecution);
        }
        let mut remaining = MAX_PYTHON_INPUT;
        write_input(file, b"{\"sessions\":[", &mut remaining).await?;
        let mut separator = b"".as_slice();
        let mut missing: BTreeSet<_> = request.span_ids.iter().cloned().collect();
        for execution in &self.executions {
            if !request.execution_ids.is_empty() && !request.execution_ids.contains(&execution.id) {
                continue;
            }
            write_input(file, separator, &mut remaining).await?;
            write_input(file, b"{\"execution\":", &mut remaining).await?;
            write_input(file, &serde_json::to_vec(execution)?, &mut remaining).await?;
            write_input(file, b",\"parts\":[", &mut remaining).await?;
            separator = b",";
            let mut part_separator = b"".as_slice();
            let sources = self.sources(execution, &request.span_ids);
            futures_util::pin_mut!(sources);
            while let Some(source) = sources.try_next().await? {
                missing.remove(&source.part.span_id);
                let mut metadata = serde_json::to_value(&source.part)?;
                let object = metadata.as_object_mut().ok_or(Error::InvalidRequest)?;
                object.remove("content");
                object.insert("truncated".into(), false.into());
                let encoded = serde_json::to_vec(&metadata)?;
                write_input(file, part_separator, &mut remaining).await?;
                write_input(file, &encoded[..encoded.len() - 1], &mut remaining).await?;
                write_input(file, b",\"content\":\"", &mut remaining).await?;
                part_separator = b",";
                let chunks = self.chunks(&source, 0);
                futures_util::pin_mut!(chunks);
                while let Some(chunk) = chunks.try_next().await? {
                    let encoded = serde_json::to_vec(&chunk.content)?;
                    write_input(file, &encoded[1..encoded.len() - 1], &mut remaining).await?;
                }
                write_input(file, b"\"}", &mut remaining).await?;
            }
            write_input(
                file,
                if self.partial(execution) {
                    b"],\"partial\":true}"
                } else {
                    b"],\"partial\":false}"
                },
                &mut remaining,
            )
            .await?;
        }
        if !missing.is_empty() {
            return Err(Error::UnknownPythonSpan);
        }
        write_input(file, b"],\"reviews\":[", &mut remaining).await?;
        let mut separator = b"".as_slice();
        for review in &self.reviews {
            if !request.execution_ids.is_empty()
                && !request.execution_ids.contains(&review.execution_id)
            {
                continue;
            }
            write_input(file, separator, &mut remaining).await?;
            write_input(file, &serde_json::to_vec(review)?, &mut remaining).await?;
            separator = b",";
        }
        write_input(file, b"]}", &mut remaining).await?;
        file.flush().await?;
        Ok(())
    }
}

async fn write_input(
    file: &mut tokio::fs::File,
    bytes: &[u8],
    remaining: &mut usize,
) -> Result<(), Error> {
    *remaining = remaining
        .checked_sub(bytes.len())
        .ok_or(Error::PythonInputTooLarge)?;
    file.write_all(bytes).await?;
    Ok(())
}

pub fn character_range(text: &str, start: usize, end: Option<usize>) -> String {
    text.chars()
        .skip(start)
        .take(
            end.map(|end| end.saturating_sub(start))
                .unwrap_or(usize::MAX),
        )
        .collect()
}

pub fn limited(value: Value) -> Result<Value, Error> {
    if serde_json::to_vec(&value)?.len() > MAX_TOOL_BYTES {
        return Err(Error::TooLarge);
    }
    Ok(value)
}
