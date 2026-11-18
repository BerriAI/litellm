//! What Python prices of a realtime or Live session through `NativeModelCostMap.price_session`: the session's
//! tokens, its `response.done` turns and the per-second, per-message, translation and transcription parts beside
//! them, in one crossing. Python reads the events and the logging object into quantities and resolves each candidate
//! model to the row `get_model_info` found, and each part names its rows by their index among the rows passed beside
//! the document

use litellm_cost::query::{
    AdjustmentPolicy, Basis, Candidate, ClipUsage, Header, Metered, PriceQuery, PriceSource,
    SessionQuery, SessionTokens, SessionTurn, TokenCandidate, TokenUsage, TranscriptionClip,
    Translation,
};
use litellm_cost::row::RowRef;
use litellm_host_python::from_py_argument;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::Deserialize;

use super::kinds::{outcome, per_call_rows};
use crate::model_cost::CostSnapshot;

#[derive(Deserialize)]
struct SessionDocument {
    #[serde(default)]
    transcription: Vec<ClipDocument>,
    #[serde(default)]
    translation: Option<TranslationDocument>,
    #[serde(default)]
    live_voice: Option<MeteredDocument>,
    #[serde(default)]
    audio_duration: Option<MeteredDocument>,
    #[serde(default)]
    text_messages: Option<MeteredDocument>,
    #[serde(default)]
    tokens: Option<TokensDocument>,
    #[serde(default)]
    turns: Vec<TurnDocument>,
    #[serde(default)]
    data_residency: Option<String>,
}

#[derive(Deserialize)]
struct TokensDocument {
    usage: TokenUsage,
    candidates: Vec<CandidateDocument>,
}

#[derive(Deserialize)]
struct CandidateDocument {
    candidate: Candidate,
    row: usize,
    declares: bool,
}

#[derive(Deserialize)]
struct TurnDocument {
    candidate: Candidate,
    usage: TokenUsage,
    #[serde(default)]
    row: Option<usize>,
}

#[derive(Deserialize)]
struct MeteredDocument {
    quantity: f64,
    rows: Vec<usize>,
}

#[derive(Deserialize)]
struct ClipDocument {
    rows: Vec<usize>,
    usage: ClipUsage,
}

#[derive(Deserialize)]
struct TranslationDocument {
    input_seconds: f64,
    output_seconds: f64,
    rows: Vec<usize>,
}

/// Prices a session's parts beside its tokens, a receipt with one part each
pub(crate) fn price_session(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    rows: &[Bound<'_, PyMapping>],
) -> PyResult<Py<PyAny>> {
    let document: SessionDocument = from_py_argument(document)?;
    let (source, refs) = per_call_rows(snapshot, rows)?;
    let named = |indexes: &[usize]| -> Vec<RowRef> {
        indexes
            .iter()
            .filter_map(|index| refs.get(*index).cloned())
            .collect()
    };
    let metered = |used: Option<MeteredDocument>| {
        used.map(|used| Metered {
            quantity: used.quantity,
            rows: named(&used.rows),
        })
    };
    let query = PriceQuery::Session(Box::new(SessionQuery {
        header: Header {
            candidates: Vec::new(),
            source: PriceSource::Rows { rows: refs.clone() },
            basis: Basis {
                service_tier: Default::default(),
                data_residency: super::query::data_residency(document.data_residency.as_deref()),
                vertex_location: None,
                at: jiff::Timestamp::now(),
            },
            adjustments: AdjustmentPolicy::None,
            reported: None,
        },
        transcription: document
            .transcription
            .into_iter()
            .map(|clip| TranscriptionClip {
                rows: named(&clip.rows),
                usage: clip.usage,
            })
            .collect(),
        translation: document.translation.map(|used| Translation {
            input_seconds: used.input_seconds,
            output_seconds: used.output_seconds,
            rows: named(&used.rows),
        }),
        live_voice: metered(document.live_voice),
        audio_duration: metered(document.audio_duration),
        text_messages: metered(document.text_messages),
        tokens: document.tokens.map(|tokens| SessionTokens {
            usage: tokens.usage,
            candidates: tokens
                .candidates
                .into_iter()
                .filter_map(|offered| {
                    Some(TokenCandidate {
                        candidate: offered.candidate,
                        row: refs.get(offered.row)?.clone(),
                        declares: offered.declares,
                    })
                })
                .collect(),
        }),
        turns: document
            .turns
            .into_iter()
            .map(|turn| SessionTurn {
                candidate: turn.candidate,
                usage: turn.usage,
                row: turn.row.and_then(|index| refs.get(index).cloned()),
            })
            .collect(),
    }));
    outcome(py, litellm_cost::price(&source, &query))
}
