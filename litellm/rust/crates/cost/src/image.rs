//! The `image` kind, a port of main's flat image calculators: `default_image_cost_calculator` (`cost_calculator.py`),
//! the per-image calculators of AI/ML, CometAPI, Recraft and RunwayML, and the DashScope one, which adds billable
//! input images. Which calculator a call takes is the route's choice, as `route_image_generation_cost_calculator`
//! makes it, so the query names it. The header's `Rows` are the deployment's own prices, read ahead of the catalog
//! row. The per-image fallback of the Bedrock, Gemini and Vertex AI calculators is the per-image calculator, and a
//! call served at a Vertex AI regional endpoint bills the priced row's regional endpoint uplift on its images. The
//! xAI calculator bills the `cost_in_usd_ticks` the response reported, else the deployment's flat image price, else
//! the price of the requested resolution, and the Azure AI one bills a positive per-image price or hands a row priced
//! per pixel to the default calculator. Calls main bills from token usage are not ported, and their routes keep
//! main's calculators. Discount and margin never apply

use crate::price::{Channel, ImageSize, InputUnit, Meter, OutputUnit};
use crate::priced::{Line, Quantity, Side};
use crate::query::{Header, Kind, PriceSource};
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::reported::{self, ReportedQuery};
use crate::row::{LoweredRow, RowRef, RowSource};
use crate::units::{self, Billed};

const INPUT_IMAGE: Meter = Meter::Input(InputUnit::Image);
const OUTPUT_IMAGE: Meter = Meter::Output(OutputUnit::Image(None));
const INPUT_PIXEL: Meter = Meter::Input(InputUnit::Pixel);

/// One image call
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct ImageQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    pub calculator: ImageCalculator,
    /// Images the call returned, or the `n` it asked for where main bills that
    pub images: u64,
    /// Width times height times images, for a row priced per pixel
    #[cfg_attr(feature = "serde", serde(default))]
    pub pixels: u64,
    /// DashScope's `billable_input_image_count`
    #[cfg_attr(feature = "serde", serde(default))]
    pub input_images: u64,
    /// The `output_cost_per_image_<size>` label the xAI calculator reads, such as `1K`
    #[cfg_attr(feature = "serde", serde(default))]
    pub size: Option<Box<str>>,
}

/// Which of main's image calculators prices the call
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ImageCalculator {
    /// `default_image_cost_calculator`: the first table that sets any of `input_cost_per_image`,
    /// `output_cost_per_image` or `input_cost_per_pixel`, read in that order
    Default,
    /// `output_cost_per_image` per image, the deployment's rate ahead of the catalog's
    PerImage,
    /// DashScope: a required `output_cost_per_image` per image plus `input_cost_per_image` per billable input image
    PerImageAndInput,
    /// xAI: the reported cost, else the first `Rows` table's `input_cost_per_image` or `output_cost_per_image`, else
    /// the first table's `output_cost_per_image_<size>`
    Xai,
    /// Azure AI: the catalog row's `output_cost_per_image` when above zero. A row priced per pixel, by its own
    /// `input_cost_per_pixel` or else the `Rows` table's, is `NotPorted`, which the default calculator prices
    AzureAi,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &ImageQuery) -> PriceOutcome {
    match query.calculator {
        ImageCalculator::Xai => xai(source, query),
        ImageCalculator::AzureAi => azure_ai(source, query),
        ImageCalculator::Default
        | ImageCalculator::PerImage
        | ImageCalculator::PerImageAndInput => flat(source, query),
    }
}

fn given(header: &Header) -> &[RowRef] {
    match &header.source {
        PriceSource::Rows { rows } => rows,
        _ => &[],
    }
}

fn xai<S: RowSource>(source: &S, query: &ImageQuery) -> PriceOutcome {
    if let Some(reported) = &query.header.reported {
        return reported::price(
            source,
            &ReportedQuery {
                header: query.header.clone(),
                reported: reported.clone(),
                split_by: None,
            },
        );
    }
    let layers = units::layers(source, &query.header);
    let Some((first, _)) = layers.first() else {
        return units::no_row(&query.header);
    };
    let deployment = given(&query.header)
        .first()
        .and_then(|at| Some((at, source.row(at)?)));
    let flat = deployment.and_then(|(at, row)| {
        [INPUT_IMAGE, OUTPUT_IMAGE].into_iter().find_map(|meter| {
            let rate = units::listed(row, &meter, Channel::Online)?;
            Some((at.clone(), meter, rate))
        })
    });
    let sized = query
        .size
        .as_deref()
        .and_then(ImageSize::new)
        .map(|size| Meter::Output(OutputUnit::Image(Some(size))));
    let at_size = || {
        let meter = sized.clone()?;
        layers.iter().find_map(|(at, row)| {
            let rate = units::listed(*row, &meter, Channel::Online)?;
            Some((at.clone(), meter.clone(), rate))
        })
    };
    let Some((at, meter, rate)) = flat.or_else(at_size) else {
        return Unpriced::MissingRates {
            row: first.clone(),
            missing: [INPUT_IMAGE, OUTPUT_IMAGE]
                .into_iter()
                .chain(sized)
                .collect(),
        }
        .into();
    };
    let line = units::line(meter, Quantity::Count(query.images), rate, Side::Completion);
    priced(source, query, &at, vec![line])
}

fn azure_ai<S: RowSource>(source: &S, query: &ImageQuery) -> PriceOutcome {
    let Some((at, row)) = query
        .header
        .candidates
        .iter()
        .find_map(|candidate| source.resolve(candidate))
        .and_then(|at| Some((at.clone(), source.row(&at)?)))
    else {
        return units::no_row(&query.header);
    };
    let per_image =
        units::listed(row, &OUTPUT_IMAGE, Channel::Online).filter(|rate| rate.get() > 0.0);
    if let Some(rate) = per_image {
        let line = units::line(
            OUTPUT_IMAGE,
            Quantity::Count(query.images),
            rate,
            Side::Completion,
        );
        return priced(source, query, &at, vec![line]);
    }
    let pixel = units::listed(row, &INPUT_PIXEL, Channel::Online).or_else(|| {
        given(&query.header)
            .iter()
            .filter_map(|at| source.row(at))
            .find_map(|row| units::listed(row, &INPUT_PIXEL, Channel::Online))
    });
    match pixel.filter(|rate| rate.get() > 0.0) {
        Some(_) => Unpriced::NotPorted(Kind::Image).into(),
        None => Unpriced::MissingRates {
            row: at,
            missing: vec![OUTPUT_IMAGE, INPUT_PIXEL],
        }
        .into(),
    }
}

fn flat<S: RowSource>(source: &S, query: &ImageQuery) -> PriceOutcome {
    let layers = units::layers(source, &query.header);
    let Some((first, _)) = layers.first() else {
        return units::no_row(&query.header);
    };
    let read = |meter: &Meter| {
        layers.iter().find_map(|(at, row)| {
            units::listed(*row, meter, Channel::Online).map(|rate| (at.clone(), rate))
        })
    };
    let images = Quantity::Count(query.images);
    let charges: Vec<(Meter, Quantity, Side)> = match query.calculator {
        ImageCalculator::Default => vec![
            (INPUT_IMAGE, images, Side::Prompt),
            (OUTPUT_IMAGE, images, Side::Completion),
            (INPUT_PIXEL, Quantity::Count(query.pixels), Side::Prompt),
        ],
        ImageCalculator::PerImage
        | ImageCalculator::PerImageAndInput
        | ImageCalculator::Xai
        | ImageCalculator::AzureAi => vec![(OUTPUT_IMAGE, images, Side::Completion)],
    };
    let uplift = |row: &S::Row| {
        row.adjustments()
            .regional_endpoint(query.header.basis.vertex_location.as_deref())
    };
    let first_priced = layers.iter().find_map(|(at, row)| {
        charges.iter().find_map(|(meter, quantity, side)| {
            units::listed(*row, meter, Channel::Online).map(|rate| {
                let line = units::line(meter.clone(), *quantity, rate, *side);
                let amount = uplift(row).map_or(line.amount, |uplift| line.amount * uplift);
                (at.clone(), Line { amount, ..line })
            })
        })
    });
    let Some((at, line)) = first_priced else {
        return Unpriced::MissingRates {
            row: first.clone(),
            missing: charges.into_iter().map(|(meter, _, _)| meter).collect(),
        }
        .into();
    };
    let input = match query.calculator {
        ImageCalculator::PerImageAndInput => read(&INPUT_IMAGE).map(|(_, rate)| {
            units::line(
                INPUT_IMAGE,
                Quantity::Count(query.input_images),
                rate,
                Side::Prompt,
            )
        }),
        ImageCalculator::Default
        | ImageCalculator::PerImage
        | ImageCalculator::Xai
        | ImageCalculator::AzureAi => None,
    };
    priced(
        source,
        query,
        &at,
        [line].into_iter().chain(input).collect(),
    )
}

fn priced<S: RowSource>(
    source: &S,
    query: &ImageQuery,
    at: &RowRef,
    lines: Vec<Line>,
) -> PriceOutcome {
    let receipt = units::receipt(source, &query.header, None, Billed::of(lines, Vec::new()));
    Receipt {
        row: Some(at.clone()),
        candidate: query.header.candidates.first().cloned(),
        ..receipt
    }
    .into()
}

#[cfg(test)]
mod parity {
    //! Parity with main's `default_image_cost_calculator` and the per-image calculators. `fixtures/image_parity.json` was
    //! captured once from main's Python: each case holds the row the calculator read, the images and pixels it billed and
    //! main's total

    use rstest::rstest;

    use super::{ImageCalculator, ImageQuery, price};
    use crate::testing::{TestSource, fixture_case, header, priced, units_row};

    const FIXTURE: &str = include_str!("../fixtures/image_parity.json");

    #[rstest]
    #[case::azure_hd_tall("azure_hd_tall", ImageCalculator::Default)]
    #[case::azure_standard_bare_model("azure_standard_bare_model", ImageCalculator::Default)]
    #[case::aiml_by_name("aiml_by_name", ImageCalculator::Default)]
    #[case::recraft_by_name("recraft_by_name", ImageCalculator::Default)]
    #[case::recraft("recraft", ImageCalculator::PerImage)]
    #[case::aiml("aiml", ImageCalculator::PerImage)]
    #[case::runwayml("runwayml", ImageCalculator::PerImage)]
    #[case::dashscope_input_images("dashscope_input_images", ImageCalculator::PerImageAndInput)]
    fn an_image_call_costs_what_main_charged(
        #[case] name: &str,
        #[case] calculator: ImageCalculator,
    ) {
        let case = fixture_case(FIXTURE, name);
        let model = case["model"].as_str().unwrap();
        let source = TestSource::with_row(model, units_row(&case["row"]));
        let query = ImageQuery {
            header: header(case["provider"].as_str().unwrap(), model),
            calculator,
            images: case["images"].as_u64().unwrap(),
            pixels: case["pixels"].as_u64().unwrap_or(0),
            input_images: case["input_images"].as_u64().unwrap_or(0),
            size: None,
        };
        assert_eq!(
            Some(priced(price(&source, &query)).total),
            case["expected"].as_f64()
        );
    }
}

#[cfg(test)]
mod provider_parity {
    //! Parity with main's xAI and Azure AI image calculators. `fixtures/image_provider_parity.json` was captured once
    //! from main's Python: each case holds the deployment's own table, the row the calculator resolved, the images and
    //! main's total

    use rstest::rstest;
    use serde_json::Value;

    use super::{ImageCalculator, ImageQuery, price};
    use crate::query::{Header, PriceSource, ReportSource, ReportedCost};
    use crate::receipt::{PriceOutcome, Unpriced};
    use crate::testing::{TestSource, fixture_case, header, row_ref, units_row};

    const FIXTURE: &str = include_str!("../fixtures/image_provider_parity.json");
    const MODEL: &str = "image-model";

    fn xai_query(case: &Value) -> (TestSource, ImageQuery) {
        let catalog = TestSource::with_row(MODEL, units_row(&case["row"]));
        let (source, rows) = match case["deployment"].is_object() {
            true => (
                catalog.and_row("deployment", units_row(&case["deployment"])),
                vec![row_ref("deployment")],
            ),
            false => (catalog, Vec::new()),
        };
        let base = header("xai", MODEL);
        let query = ImageQuery {
            header: Header {
                source: match rows.is_empty() {
                    true => PriceSource::Catalog,
                    false => PriceSource::Rows { rows },
                },
                reported: case["reported_usd"].as_f64().map(|usd| ReportedCost {
                    source: ReportSource::CostInUsdTicks,
                    usd,
                }),
                ..base
            },
            calculator: ImageCalculator::Xai,
            images: case["images"].as_u64().unwrap(),
            pixels: 0,
            input_images: 0,
            size: case["size"].as_str().map(Into::into),
        };
        (source, query)
    }

    fn total(outcome: PriceOutcome) -> f64 {
        match outcome {
            PriceOutcome::Priced(receipt) => receipt.total,
            PriceOutcome::Unpriced(Unpriced::MissingRates { .. }) => 0.0,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    #[rstest]
    #[case::reported_ticks("xai_reported_ticks")]
    #[case::reported_ignored_for_deployment_price("xai_reported_ignored_for_deployment_price")]
    #[case::resolution_1k("xai_resolution_1k")]
    #[case::resolution_2k("xai_resolution_2k")]
    #[case::deployment_input_image_first("xai_deployment_input_image_first")]
    #[case::deployment_sized("xai_deployment_sized")]
    fn an_xai_image_call_costs_what_main_charged(#[case] name: &str) {
        let case = fixture_case(FIXTURE, name);
        let (source, query) = xai_query(&case);
        assert_eq!(
            Some(total(price(&source, &query))),
            case["expected"].as_f64()
        );
    }

    #[rstest]
    #[case::per_image("azure_ai_per_image")]
    #[case::deployment_per_image("azure_ai_deployment_per_image")]
    #[case::zero_per_image_unpriced("azure_ai_zero_per_image_unpriced")]
    fn an_azure_ai_image_call_costs_what_main_charged(#[case] name: &str) {
        let case = fixture_case(FIXTURE, name);
        let resolved = Value::Object(
            case["row"]
                .as_object()
                .unwrap()
                .iter()
                .chain(case["deployment"].as_object().into_iter().flatten())
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let source = TestSource::with_row(MODEL, units_row(&resolved));
        let query = ImageQuery {
            header: header("azure_ai", MODEL),
            calculator: ImageCalculator::AzureAi,
            images: case["images"].as_u64().unwrap(),
            pixels: 0,
            input_images: 0,
            size: None,
        };
        assert_eq!(
            Some(total(price(&source, &query))),
            case["expected"].as_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::money::Multiplier;
    use crate::price::{Adjustments, ServiceTier, Uplift};
    use crate::query::{Basis, PriceSource};
    use crate::testing::{
        BASE, TestRow, TestSource, header, priced as receipt_of, prices, row_ref,
    };

    fn row(rates: &[(Meter, f64)]) -> TestRow {
        TestRow {
            prices: prices(
                &rates
                    .iter()
                    .map(|(meter, rate)| (meter.clone(), ServiceTier::Standard, BASE, *rate))
                    .collect::<Vec<_>>(),
            ),
            ..TestRow::default()
        }
    }

    fn query(calculator: ImageCalculator, deployment: bool) -> ImageQuery {
        let base = header("openai", "dall-e-3");
        ImageQuery {
            header: match deployment {
                true => Header {
                    source: PriceSource::Rows {
                        rows: vec![row_ref("deployment")],
                    },
                    ..base
                },
                false => base,
            },
            calculator,
            images: 2,
            pixels: 2 * 1024 * 1024,
            input_images: 3,
            size: None,
        }
    }

    fn total(source: &TestSource, image: &ImageQuery) -> f64 {
        receipt_of(price(source, image)).total
    }

    #[rstest]
    #[case::input_image_first(row(&[(INPUT_IMAGE, 0.01), (OUTPUT_IMAGE, 0.02), (INPUT_PIXEL, 1e-8)]), 0.01 * 2.0)]
    #[case::output_image_next(row(&[(OUTPUT_IMAGE, 0.02), (INPUT_PIXEL, 1e-8)]), 0.02 * 2.0)]
    #[case::pixels_last(row(&[(INPUT_PIXEL, 1e-8)]), 1e-8 * 2_097_152.0)]
    fn the_default_calculator_reads_one_rate_in_main_order(
        #[case] priced_row: TestRow,
        #[case] expected: f64,
    ) {
        let source = TestSource::with_row("dall-e-3", priced_row);
        assert_eq!(
            total(&source, &query(ImageCalculator::Default, false)),
            expected
        );
    }

    #[rstest]
    fn the_default_calculator_reads_the_deployment_table_whole_before_the_catalog() {
        let source = TestSource::with_row("dall-e-3", row(&[(INPUT_IMAGE, 0.01)]))
            .and_row("deployment", row(&[(INPUT_PIXEL, 1e-9)]));
        assert_eq!(
            total(&source, &query(ImageCalculator::Default, true)),
            1e-9 * 2_097_152.0
        );
    }

    #[rstest]
    #[case::deployment_rate(true, 0.5 * 2.0)]
    #[case::catalog_rate(false, 0.04 * 2.0)]
    fn a_per_image_calculator_bills_the_first_output_image_rate(
        #[case] deployment: bool,
        #[case] expected: f64,
    ) {
        let source = TestSource::with_row("dall-e-3", row(&[(OUTPUT_IMAGE, 0.04)]))
            .and_row("deployment", row(&[(OUTPUT_IMAGE, 0.5)]));
        assert_eq!(
            total(&source, &query(ImageCalculator::PerImage, deployment)),
            expected
        );
    }

    #[rstest]
    #[case::regional(Some("us-east5"), 0.04 * 2.0 * 1.25)]
    #[case::multi_region(Some("eu"), 0.04 * 2.0 * 1.25)]
    #[case::global(Some("global"), 0.04 * 2.0)]
    #[case::global_any_case(Some("GLOBAL"), 0.04 * 2.0)]
    #[case::no_location(None, 0.04 * 2.0)]
    fn a_regional_endpoint_bills_the_row_uplift_on_its_images(
        #[case] location: Option<&str>,
        #[case] expected: f64,
    ) {
        let uplifted = TestRow {
            adjustments: Adjustments::new([(
                Uplift::NonGlobalEndpoint,
                Multiplier::new(1.25).unwrap(),
            )]),
            ..row(&[(OUTPUT_IMAGE, 0.04)])
        };
        let source = TestSource::with_row("dall-e-3", uplifted);
        let base = query(ImageCalculator::PerImage, false);
        let located = ImageQuery {
            header: Header {
                basis: Basis {
                    vertex_location: location.map(Into::into),
                    ..base.header.basis.clone()
                },
                ..base.header.clone()
            },
            ..base
        };
        assert_eq!(total(&source, &located), expected);
    }

    #[rstest]
    #[case::own_pixel_rate(row(&[(INPUT_PIXEL, 5e-8)]), None, true)]
    #[case::shared_entry_pixel_rate(row(&[]), Some(row(&[(INPUT_PIXEL, 5e-8)])), true)]
    #[case::own_zero_pixel_rate_beats_the_shared_entry(
        row(&[(INPUT_PIXEL, 0.0)]),
        Some(row(&[(INPUT_PIXEL, 5e-8)])),
        false
    )]
    #[case::zero_per_image_rate_still_routes_pixels(
        row(&[(OUTPUT_IMAGE, 0.0), (INPUT_PIXEL, 5e-8)]),
        None,
        true
    )]
    #[case::no_rate(row(&[]), None, false)]
    fn azure_ai_hands_a_row_priced_per_pixel_to_the_default_calculator(
        #[case] resolved: TestRow,
        #[case] shared: Option<TestRow>,
        #[case] routed: bool,
    ) {
        let source = TestSource::with_row("dall-e-3", resolved);
        let (source, azure) = match shared {
            Some(shared) => (
                source.and_row("deployment", shared),
                query(ImageCalculator::AzureAi, true),
            ),
            None => (source, query(ImageCalculator::AzureAi, false)),
        };
        let outcome = price(&source, &azure);
        assert_eq!(
            matches!(
                outcome,
                PriceOutcome::Unpriced(Unpriced::NotPorted(Kind::Image))
            ),
            routed
        );
        assert!(!matches!(outcome, PriceOutcome::Priced(_)));
    }

    #[rstest]
    fn xai_bills_the_deployment_flat_price_ahead_of_its_sized_price() {
        let source = TestSource::with_row("dall-e-3", row(&[(OUTPUT_IMAGE, 0.5)])).and_row(
            "deployment",
            row(&[(sized("2K"), 0.04), (OUTPUT_IMAGE, 0.03)]),
        );
        let deployment = ImageQuery {
            size: Some("2K".into()),
            ..query(ImageCalculator::Xai, true)
        };
        assert_eq!(total(&source, &deployment), 0.03 * 2.0);
    }

    #[rstest]
    fn xai_never_bills_the_catalog_flat_price() {
        let source = TestSource::with_row("dall-e-3", row(&[(OUTPUT_IMAGE, 0.5)]));
        let catalog = ImageQuery {
            size: Some("1K".into()),
            ..query(ImageCalculator::Xai, false)
        };
        assert!(matches!(
            price(&source, &catalog),
            PriceOutcome::Unpriced(Unpriced::MissingRates { .. })
        ));
    }

    fn sized(label: &str) -> Meter {
        Meter::Output(OutputUnit::Image(ImageSize::new(label)))
    }

    #[rstest]
    fn dashscope_adds_its_billable_input_images() {
        let source = TestSource::with_row(
            "dall-e-3",
            row(&[(OUTPUT_IMAGE, 0.04), (INPUT_IMAGE, 0.001)]),
        );
        assert_eq!(
            total(&source, &query(ImageCalculator::PerImageAndInput, false)),
            0.04 * 2.0 + 0.001 * 3.0
        );
    }

    #[rstest]
    #[case::default(ImageCalculator::Default)]
    #[case::per_image(ImageCalculator::PerImage)]
    fn a_row_without_an_image_rate_is_missing_it(#[case] calculator: ImageCalculator) {
        let source = TestSource::with_row("dall-e-3", row(&[(Meter::TEXT_INPUT, 1.0)]));
        assert!(matches!(
            price(&source, &query(calculator, false)),
            PriceOutcome::Unpriced(Unpriced::MissingRates { .. })
        ));
    }
}
