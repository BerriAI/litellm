//! The kinds Python prices per unit through `NativeModelCostMap.price_units`. Each document carries the facts main's
//! calculator read off the call: the model and provider it priced, and the units the response reported. Beside it come
//! the deployment's own price tables and the row main's calculator read through `get_model_info`, which Python looks
//! up with its own `get_model_info`, so a lookup a caller patched or a model only a capability rule knows reads the
//! same row it did on main. The answer is the receipt or why the call is unpriced, and Python maps that outcome to
//! each calculator's legacy return

use litellm_callbacks_legacy_python::NativeReceipt;
use litellm_cost::query::{
    AdjustmentPolicy, Basis, BatchLineQuery, BatchUsage, Candidate, Header, ImageCalculator,
    ImageQuery, OcrQuery, PriceQuery, PriceSource, Provider, RerankQuery, SearchQuery, SpeechQuery,
    TranscriptionQuery, VectorStoreSearchQuery, VideoQuery,
};
use litellm_cost::receipt::{PriceOutcome, Unpriced};
use litellm_cost::row::RowRef;
use litellm_host_python::from_py_argument;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::Deserialize;

use crate::model_cost::{CostSnapshot, row_fields};

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum UnitsDocument {
    Rerank {
        #[serde(flatten)]
        named: Named,
        #[serde(default)]
        search_units: Option<u64>,
        #[serde(default)]
        total_tokens: Option<u64>,
    },
    VectorStoreSearch {
        #[serde(flatten)]
        named: Named,
    },
    Ocr {
        #[serde(flatten)]
        named: Named,
        #[serde(default)]
        pages: Option<u64>,
        #[serde(default)]
        annotation_pages: u64,
        #[serde(default)]
        credits: Option<f64>,
        #[serde(default)]
        batch: bool,
        #[serde(default)]
        catalog_deferred: bool,
    },
    Speech {
        #[serde(flatten)]
        named: Named,
        #[serde(default)]
        characters: Option<u64>,
        #[serde(default)]
        generations: Option<u64>,
    },
    Transcription {
        #[serde(flatten)]
        named: Named,
        seconds: f64,
        #[serde(default)]
        tokens: bool,
    },
    Image {
        #[serde(flatten)]
        named: Named,
        calculator: ImageCalculator,
        images: u64,
        #[serde(default)]
        pixels: u64,
        #[serde(default)]
        input_images: u64,
    },
    Video {
        #[serde(flatten)]
        named: Named,
        #[serde(default)]
        seconds: Option<f64>,
        #[serde(default)]
        resolution: Option<Box<str>>,
        videos: u64,
        #[serde(default)]
        input_images: u64,
        /// `tables` holds the deployment's own table first, which replaces the catalog row
        #[serde(default)]
        deployment: bool,
        /// The last of `tables` is the requested model's row, read for an input image rate
        #[serde(default)]
        requested_row: bool,
    },
    BatchLine {
        #[serde(flatten)]
        named: Named,
        usage: BatchUsage,
        #[serde(default)]
        data_residency: Option<String>,
        #[serde(default)]
        vertex_location: Option<Box<str>>,
    },
    Search {
        #[serde(flatten)]
        named: Named,
        queries: u64,
        #[serde(default)]
        additional_results: u64,
        #[serde(default)]
        adjust: Option<AdjustFacts>,
    },
}

/// The discount and margin key of a call main adjusts
#[derive(Deserialize)]
struct AdjustFacts {
    #[serde(default)]
    provider: Option<Box<str>>,
    #[serde(default)]
    model: Option<Box<str>>,
    #[serde(default)]
    region: Option<Box<str>>,
}

/// The model and provider main's calculator priced under
#[derive(Deserialize)]
struct Named {
    model: String,
    #[serde(default)]
    provider: Option<String>,
}

/// Why `price_units` priced nothing
#[pyclass(
    frozen,
    get_all,
    name = "NativeUnpriced",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeUnpriced {
    /// `no_row`, `missing_rates`, `missing_usage`, `catalog_deferred` or `not_ported`
    reason: &'static str,
    /// The row that lacked a rate or a quantity
    row: Option<String>,
}

/// Prices one per-unit call. `tables` are the deployment's own price tables, read ahead of `catalog` in the order
/// given, and `catalog` is the row `get_model_info` found for the call's model, `None` when it found none
pub(crate) fn price_units(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    tables: &[Bound<'_, PyMapping>],
    catalog: Option<&Bound<'_, PyMapping>>,
) -> PyResult<Py<PyAny>> {
    let document: UnitsDocument = from_py_argument(document)?;
    let (candidate, provider) = document.candidate();
    let rows = match catalog {
        Some(catalog) => snapshot.pricing_rows([], []).with_named_per_call(
            candidate.clone(),
            "get_model_info",
            row_fields(catalog)?,
            &provider,
        ),
        None => snapshot.pricing_rows([], []),
    };
    let (rows, given) = tables.iter().enumerate().try_fold(
        (rows, Vec::new()),
        |(rows, given), (index, table)| -> PyResult<_> {
            let (rows, at) =
                rows.with_per_call(&format!("table-{index}"), row_fields(table)?, &provider);
            Ok((rows, given.into_iter().chain([at]).collect::<Vec<RowRef>>()))
        },
    )?;
    let (rows_given, image_rows) = match &document {
        UnitsDocument::Video {
            deployment,
            requested_row,
            ..
        } => {
            let deployment_row = given.first().filter(|_| *deployment).cloned();
            let requested = given.last().filter(|_| *requested_row).cloned();
            (
                deployment_row.iter().cloned().collect(),
                deployment_row.into_iter().chain(requested).collect(),
            )
        }
        _ => (given, Vec::new()),
    };
    let source = match rows_given.is_empty() {
        true => PriceSource::Catalog,
        false => PriceSource::Rows { rows: rows_given },
    };
    let query = document.query(candidate, source, image_rows);
    outcome(py, litellm_cost::price(&rows, &query))
}

impl UnitsDocument {
    fn named(&self) -> &Named {
        match self {
            Self::Rerank { named, .. }
            | Self::VectorStoreSearch { named }
            | Self::Ocr { named, .. }
            | Self::Speech { named, .. }
            | Self::Transcription { named, .. }
            | Self::Search { named, .. }
            | Self::Video { named, .. }
            | Self::Image { named, .. }
            | Self::BatchLine { named, .. } => named,
        }
    }

    /// The model and provider main's calculator priced under, and the provider as written
    fn candidate(&self) -> (Candidate, String) {
        let named = self.named();
        let provider = named.provider.clone().unwrap_or_default();
        let candidate = Candidate {
            model: named.model.as_str().into(),
            provider: provider
                .parse()
                .unwrap_or_else(|_| Provider::Other(provider.clone())),
        };
        (candidate, provider)
    }

    fn query(
        self,
        candidate: Candidate,
        source: PriceSource,
        image_rows: Vec<RowRef>,
    ) -> PriceQuery {
        let (data_residency, vertex_location) = match &self {
            Self::BatchLine {
                data_residency,
                vertex_location,
                ..
            } => (
                super::query::data_residency(data_residency.as_deref()),
                vertex_location.clone(),
            ),
            _ => (None, None),
        };
        let header = Header {
            candidates: vec![candidate],
            source,
            basis: Basis {
                service_tier: Default::default(),
                data_residency,
                vertex_location,
                at: jiff::Timestamp::now(),
            },
            adjustments: match &self {
                Self::Search {
                    adjust: Some(adjust),
                    ..
                } => AdjustmentPolicy::Apply {
                    provider: adjust.provider.clone(),
                    model: adjust.model.clone(),
                    region: adjust.region.clone(),
                },
                _ => AdjustmentPolicy::None,
            },
            reported: None,
        };
        match self {
            Self::Rerank {
                search_units,
                total_tokens,
                ..
            } => PriceQuery::Rerank(Box::new(RerankQuery {
                header,
                search_units,
                total_tokens,
            })),
            Self::VectorStoreSearch { .. } => {
                PriceQuery::VectorStoreSearch(Box::new(VectorStoreSearchQuery {
                    header,
                    queries: 1,
                }))
            }
            Self::Ocr {
                pages,
                annotation_pages,
                credits,
                batch,
                catalog_deferred,
                ..
            } => PriceQuery::Ocr(Box::new(OcrQuery {
                header,
                pages,
                annotation_pages,
                credits,
                batch,
                catalog_deferred,
            })),
            Self::Speech {
                characters,
                generations,
                ..
            } => PriceQuery::Speech(Box::new(SpeechQuery {
                header,
                characters,
                generations,
            })),
            Self::Transcription {
                seconds, tokens, ..
            } => PriceQuery::Transcription(Box::new(TranscriptionQuery {
                header,
                seconds,
                tokens,
            })),
            Self::Image {
                calculator,
                images,
                pixels,
                input_images,
                ..
            } => PriceQuery::Image(Box::new(ImageQuery {
                header,
                calculator,
                images,
                pixels,
                input_images,
            })),
            Self::Video {
                seconds,
                resolution,
                videos,
                input_images,
                ..
            } => PriceQuery::Video(Box::new(VideoQuery {
                header,
                seconds,
                resolution,
                videos,
                input_images,
                input_image_rows: image_rows,
            })),
            Self::Search {
                queries,
                additional_results,
                ..
            } => PriceQuery::Search(Box::new(SearchQuery {
                header,
                queries,
                additional_results,
            })),
            Self::BatchLine { usage, .. } => {
                PriceQuery::BatchLine(Box::new(BatchLineQuery { header, usage }))
            }
        }
    }
}

/// One call of a `price_many` request and the index of its row among the rows passed beside it, `None` when no row
/// prices it
#[derive(Deserialize)]
struct ManyDocument {
    #[serde(flatten)]
    document: UnitsDocument,
    #[serde(default)]
    row: Option<usize>,
}

/// Prices many per-unit calls in one crossing, such as the lines of a batch output file. Each row is lowered once
/// however many calls name it, and each call is answered as `price_units` would answer it alone with its row as the
/// row `get_model_info` found
pub(crate) fn price_many(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    documents: &Bound<'_, PyAny>,
    rows: &[Bound<'_, PyMapping>],
) -> PyResult<Vec<Py<PyAny>>> {
    let documents: Vec<ManyDocument> = from_py_argument(documents)?;
    let (source, refs) = rows.iter().enumerate().try_fold(
        (snapshot.pricing_rows([], []), Vec::new()),
        |(source, refs), (index, row)| -> PyResult<_> {
            let fields = row_fields(row)?;
            let provider = fields
                .get("litellm_provider")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let (source, at) = source.with_per_call(&format!("row-{index}"), fields, &provider);
            Ok((
                source,
                refs.into_iter().chain([at]).collect::<Vec<RowRef>>(),
            ))
        },
    )?;
    let queries: Vec<PriceQuery> = documents
        .into_iter()
        .map(|many| {
            let (candidate, _) = many.document.candidate();
            let source = match many.row.and_then(|index| refs.get(index)) {
                Some(row) => PriceSource::Row {
                    row: row.clone(),
                    breakdown_by_name: false,
                },
                None => PriceSource::Rows { rows: Vec::new() },
            };
            many.document.query(candidate, source, Vec::new())
        })
        .collect();
    litellm_cost::price_many(&source, &queries)
        .into_iter()
        .map(|priced| outcome(py, priced))
        .collect()
}

pub(super) fn outcome(py: Python<'_>, outcome: PriceOutcome) -> PyResult<Py<PyAny>> {
    let unpriced = |reason, row: Option<RowRef>| NativeUnpriced {
        reason,
        row: row.map(|row| row.key.into_string()),
    };
    let answer = match outcome {
        PriceOutcome::Priced(receipt) => {
            return Ok(Py::new(py, NativeReceipt::new(py, &receipt)?)?.into_any());
        }
        PriceOutcome::Unpriced(Unpriced::NoRow { .. }) => unpriced("no_row", None),
        PriceOutcome::Unpriced(Unpriced::MissingRates { row, .. }) => {
            unpriced("missing_rates", Some(row))
        }
        PriceOutcome::Unpriced(Unpriced::MissingUsage { row, .. }) => {
            unpriced("missing_usage", row)
        }
        PriceOutcome::Unpriced(Unpriced::CatalogDeferred) => unpriced("catalog_deferred", None),
        PriceOutcome::Unpriced(Unpriced::NotPorted(_)) => unpriced("not_ported", None),
    };
    Ok(Py::new(py, answer)?.into_any())
}
