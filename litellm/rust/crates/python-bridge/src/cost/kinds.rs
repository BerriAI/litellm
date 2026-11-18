//! The kinds Python prices per unit through `NativeModelCostMap.price_units`. Each document carries the facts main's
//! calculator read off the call: the model and provider it priced, and the units the response reported. Beside it come
//! the deployment's own price tables and the row main's calculator read through `get_model_info`, which Python looks
//! up with its own `get_model_info`, so a lookup a caller patched or a model only a capability rule knows reads the
//! same row it did on main. The answer is the receipt or why the call is unpriced, and Python maps that outcome to
//! each calculator's legacy return

use litellm_callbacks_legacy_python::NativeReceipt;
use litellm_cost::query::{
    AdjustmentPolicy, Basis, Candidate, Header, OcrQuery, PriceQuery, PriceSource, Provider,
    RerankQuery, VectorStoreSearchQuery,
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
    let named = match &document {
        UnitsDocument::Rerank { named, .. }
        | UnitsDocument::VectorStoreSearch { named }
        | UnitsDocument::Ocr { named, .. } => named,
    };
    let provider = named.provider.as_deref().unwrap_or_default();
    let candidate = Candidate {
        model: named.model.as_str().into(),
        provider: provider
            .parse()
            .unwrap_or_else(|_| Provider::Other(provider.to_owned())),
    };
    let rows = match catalog {
        Some(catalog) => snapshot.pricing_rows([], []).with_named_per_call(
            candidate.clone(),
            "get_model_info",
            row_fields(catalog)?,
            provider,
        ),
        None => snapshot.pricing_rows([], []),
    };
    let (rows, given) = tables.iter().enumerate().try_fold(
        (rows, Vec::new()),
        |(rows, given), (index, table)| -> PyResult<_> {
            let (rows, at) =
                rows.with_per_call(&format!("table-{index}"), row_fields(table)?, provider);
            Ok((rows, given.into_iter().chain([at]).collect::<Vec<RowRef>>()))
        },
    )?;
    let header = Header {
        candidates: vec![candidate],
        source: match given.is_empty() {
            true => PriceSource::Catalog,
            false => PriceSource::Rows { rows: given },
        },
        basis: Basis {
            service_tier: Default::default(),
            data_residency: None,
            vertex_location: None,
            at: jiff::Timestamp::now(),
        },
        adjustments: AdjustmentPolicy::None,
        reported: None,
    };
    let query = match document {
        UnitsDocument::Rerank {
            search_units,
            total_tokens,
            ..
        } => PriceQuery::Rerank(Box::new(RerankQuery {
            header,
            search_units,
            total_tokens,
        })),
        UnitsDocument::VectorStoreSearch { .. } => {
            PriceQuery::VectorStoreSearch(Box::new(VectorStoreSearchQuery { header, queries: 1 }))
        }
        UnitsDocument::Ocr {
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
    };
    outcome(py, litellm_cost::price(&rows, &query))
}

fn outcome(py: Python<'_>, outcome: PriceOutcome) -> PyResult<Py<PyAny>> {
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
            unpriced("missing_usage", Some(row))
        }
        PriceOutcome::Unpriced(Unpriced::CatalogDeferred) => unpriced("catalog_deferred", None),
        PriceOutcome::Unpriced(Unpriced::NotPorted(_)) => unpriced("not_ported", None),
    };
    Ok(Py::new(py, answer)?.into_any())
}
