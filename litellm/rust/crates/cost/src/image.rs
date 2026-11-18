//! The `image` kind, a port of main's flat image calculators: `default_image_cost_calculator` (`cost_calculator.py`),
//! the per-image calculators of AI/ML, CometAPI, Recraft and RunwayML, and the DashScope one, which adds billable
//! input images. Which calculator a call takes is the route's choice, as `route_image_generation_cost_calculator`
//! makes it, so the query names it. The header's `Rows` are the deployment's own prices, read ahead of the catalog
//! row. Calls main bills from token usage, and the Bedrock, fal, xAI, Azure AI, Gemini and Vertex AI calculators, are
//! not ported, and their routes keep main's calculators. Discount and margin never apply

use crate::price::{Channel, InputUnit, Meter, OutputUnit};
use crate::priced::{Line, Quantity, Side};
use crate::query::Header;
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::{RowRef, RowSource};
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
}

pub(crate) fn price<S: RowSource>(source: &S, query: &ImageQuery) -> PriceOutcome {
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
        ImageCalculator::PerImage | ImageCalculator::PerImageAndInput => {
            vec![(OUTPUT_IMAGE, images, Side::Completion)]
        }
    };
    let first_priced = layers.iter().find_map(|(at, row)| {
        charges.iter().find_map(|(meter, quantity, side)| {
            units::listed(*row, meter, Channel::Online).map(|rate| {
                (
                    at.clone(),
                    units::line(meter.clone(), *quantity, rate, *side),
                )
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
        ImageCalculator::Default | ImageCalculator::PerImage => None,
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
        };
        assert_eq!(
            Some(priced(price(&source, &query)).total),
            case["expected"].as_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::query::PriceSource;
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
