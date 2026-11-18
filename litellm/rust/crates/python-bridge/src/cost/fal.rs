//! fal.ai calls Python prices through `NativeModelCostMap.price_fal`: an image response (`cost_calculator`) or a queue
//! submission (`fal_ai_passthrough_cost`). Python reads the facts off the response and request and passes the
//! `fal_ai/<model>` row as `catalog`, and Model Catalog picks each image's size keyed row here

use litellm_cost::query::{
    AdjustmentPolicy, Basis, Candidate, FalCall, FalImage, FalQuery, Header, PriceQuery,
    PriceSource, Provider,
};
use litellm_cost::row::RowRef;
use litellm_host_python::from_py_argument;
use litellm_model_catalog::priced_ref;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::Deserialize;

use super::kinds::outcome;
use crate::model_cost::{CostSnapshot, row_fields};

const PROVIDER: &str = "fal_ai";

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FalDocument {
    /// `tables` holds the deployment's own prices first when `deployment` is set, then the row
    /// `resolve_image_model_info` found when `fallback` is set
    FalImage {
        model: String,
        quality: String,
        images: Vec<ImageFacts>,
        #[serde(default)]
        billable_units: Option<f64>,
        #[serde(default)]
        resolution: Option<Box<str>>,
        #[serde(default)]
        deployment: bool,
        #[serde(default)]
        fallback: bool,
    },
    FalSubmit {
        model: String,
        #[serde(default)]
        seconds: Option<f64>,
        #[serde(default)]
        clip_resolution: Option<Box<str>>,
        #[serde(default)]
        image_resolution: Option<Box<str>>,
        images: u64,
    },
}

/// One returned image: the size its keyed row is picked by, and its own pixels when the response reported them
#[derive(Deserialize)]
struct ImageFacts {
    target: (u64, u64),
    #[serde(default)]
    pixels: Option<u64>,
}

pub(crate) fn price_fal(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    tables: &[Bound<'_, PyMapping>],
    catalog: Option<&Bound<'_, PyMapping>>,
) -> PyResult<Py<PyAny>> {
    let document: FalDocument = from_py_argument(document)?;
    let model = match &document {
        FalDocument::FalImage { model, .. } | FalDocument::FalSubmit { model, .. } => model,
    };
    let candidate = Candidate {
        model: model.as_str().into(),
        provider: Provider::Other(PROVIDER.into()),
    };
    let keyed: Vec<Option<RowRef>> = match &document {
        FalDocument::FalImage {
            model,
            quality,
            images,
            ..
        } => images
            .iter()
            .map(|image| {
                snapshot
                    .fal_sized_row(model, quality, image.target)
                    .map(|row| priced_ref(&row))
            })
            .collect(),
        FalDocument::FalSubmit { .. } => Vec::new(),
    };
    let fetched = keyed
        .iter()
        .flatten()
        .filter_map(|at| snapshot.row_at(&at.key));
    let rows = snapshot.pricing_rows([], fetched);
    let rows = match catalog {
        Some(catalog) => {
            rows.with_named_per_call(candidate.clone(), "catalog", row_fields(catalog)?, PROVIDER)
        }
        None => rows,
    };
    let (rows, given) = tables.iter().enumerate().try_fold(
        (rows, Vec::new()),
        |(rows, given), (index, table)| -> PyResult<_> {
            let (rows, at) =
                rows.with_per_call(&format!("table-{index}"), row_fields(table)?, PROVIDER);
            Ok((rows, given.into_iter().chain([at]).collect::<Vec<RowRef>>()))
        },
    )?;
    let header = |deployment: Option<RowRef>| Header {
        candidates: vec![candidate.clone()],
        source: match deployment {
            Some(row) => PriceSource::Rows { rows: vec![row] },
            None => PriceSource::Catalog,
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
        FalDocument::FalImage {
            images,
            billable_units,
            resolution,
            deployment,
            fallback,
            ..
        } => FalQuery {
            header: header(given.first().filter(|_| deployment).cloned()),
            call: FalCall::Image {
                images: images
                    .into_iter()
                    .zip(keyed)
                    .map(|(image, keyed)| FalImage {
                        keyed,
                        pixels: image.pixels,
                    })
                    .collect(),
                billable_units,
                resolution,
                fallback: given.last().filter(|_| fallback).cloned(),
            },
        },
        FalDocument::FalSubmit {
            seconds,
            clip_resolution,
            image_resolution,
            images,
            ..
        } => FalQuery {
            header: header(None),
            call: FalCall::Submit {
                seconds,
                clip_resolution,
                image_resolution,
                images,
            },
        },
    };
    outcome(
        py,
        litellm_cost::price(&rows, &PriceQuery::Fal(Box::new(query))),
    )
}
