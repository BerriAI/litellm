//! fal.ai, a port of `llms/fal_ai/cost_calculator.py`: `cost_calculator`, which prices an image response, and
//! `fal_ai_passthrough_cost`, which prices a queue submission from its request. The header's `Rows` are the
//! deployment's own prices and its candidate names the `fal_ai/<model>` row. Which size keyed row each image reads is
//! Model Catalog's rule (`Snapshot::fal_sized_row`), so the query carries the rows it picked. Discount and margin
//! never apply

use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, ImageSize, Meter, OutputUnit, VideoClip};
use crate::priced::{Line, Quantity, Side};
use crate::query::{Header, PriceSource};
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::{RowRef, RowSource};
use crate::units::{self, Billed};

const OUTPUT_IMAGE: Meter = Meter::Output(OutputUnit::Image(None));
const OUTPUT_PIXEL: Meter = Meter::Output(OutputUnit::Pixel);
const BILLABLE_UNIT: Meter = Meter::Output(OutputUnit::BillableUnit);
const PER_VIDEO: Meter = Meter::Output(OutputUnit::Video);
/// `FAL_PIXELS_PER_MEGAPIXEL`
const MEGAPIXEL: u64 = 1_048_576;

#[derive(Clone, Debug, PartialEq)]
pub struct FalQuery {
    pub header: Header,
    pub call: FalCall,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FalCall {
    /// `cost_calculator`
    Image {
        /// One per returned image
        images: Vec<FalImage>,
        /// The `x-fal-billable-units` header
        billable_units: Option<f64>,
        /// The request's `resolution`, read as `output_cost_per_image_<resolution>`
        resolution: Option<Box<str>>,
        /// The row `resolve_image_model_info` found, read when no other price applies. The core answers `NoRow` when
        /// it needs this row and the caller passed none
        fallback: Option<RowRef>,
    },
    /// `fal_ai_passthrough_cost`
    Submit {
        /// The request's `duration`, when it reads as a positive number of seconds
        seconds: Option<f64>,
        /// The request's `resolution` when it is text, the clip table's column
        clip_resolution: Option<Box<str>>,
        /// The request's `resolution` as `output_cost_per_image_<resolution>` reads it
        image_resolution: Option<Box<str>>,
        /// `num_images`, 1 unless the request names a positive count
        images: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FalImage {
    /// The size keyed row Model Catalog picked for the image, `None` when the model has none
    pub keyed: Option<RowRef>,
    /// The returned image's width times height, when the response reported both
    pub pixels: Option<u64>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &FalQuery) -> PriceOutcome {
    match &query.call {
        FalCall::Image {
            images,
            billable_units,
            resolution,
            fallback,
        } => image(
            source,
            &query.header,
            images,
            *billable_units,
            resolution.as_deref(),
            fallback.as_ref(),
        ),
        FalCall::Submit {
            seconds,
            clip_resolution,
            image_resolution,
            images,
        } => submit(
            source,
            &query.header,
            *seconds,
            clip_resolution.as_deref(),
            image_resolution.as_deref(),
            *images,
        ),
    }
}

fn deployment<'a, S: RowSource>(source: &'a S, header: &Header) -> Option<(RowRef, &'a S::Row)> {
    let PriceSource::Rows { rows } = &header.source else {
        return None;
    };
    let at = rows.first()?;
    Some((at.clone(), source.row(at)?))
}

fn entry<'a, S: RowSource>(source: &'a S, header: &Header) -> Option<(RowRef, &'a S::Row)> {
    let at = header
        .candidates
        .iter()
        .find_map(|candidate| source.resolve(candidate))?;
    let row = source.row(&at)?;
    Some((at, row))
}

fn sized(resolution: Option<&str>) -> Option<Meter> {
    resolution
        .and_then(ImageSize::new)
        .map(|size| Meter::Output(OutputUnit::Image(Some(size))))
}

fn image<S: RowSource>(
    source: &S,
    header: &Header,
    images: &[FalImage],
    billable_units: Option<f64>,
    resolution: Option<&str>,
    fallback: Option<&RowRef>,
) -> PriceOutcome {
    let count = Quantity::Count(images.len() as u64);
    let listed = |row: &S::Row, meter: &Meter| units::listed(row, meter, Channel::Online);
    let by_units = |row: &S::Row| {
        let units = billable_units?;
        let rate = listed(row, &BILLABLE_UNIT)?;
        Some(units::line(
            BILLABLE_UNIT,
            Quantity::Measured(units),
            rate,
            Side::Completion,
        ))
    };
    let per_image = |row: &S::Row, meter: Meter| {
        let rate = listed(row, &meter)?;
        Some(units::line(meter, count, rate, Side::Completion))
    };
    let from_deployment = deployment(source, header).and_then(|(at, row)| {
        let line = by_units(row).or_else(|| per_image(row, OUTPUT_IMAGE))?;
        Some((Some(at), vec![line]))
    });
    let from_entry = || {
        entry(source, header).and_then(|(at, row)| {
            let line = by_units(row).or_else(|| per_image(row, sized(resolution)?))?;
            Some((Some(at), vec![line]))
        })
    };
    let from_keyed = || {
        let rates: Option<Vec<(RowRef, UnitPrice)>> = images
            .iter()
            .map(|image| {
                let at = image.keyed.as_ref()?;
                Some((at.clone(), listed(source.row(at)?, &OUTPUT_IMAGE)?))
            })
            .collect();
        let rates = rates?;
        let lines = rates
            .iter()
            .map(|(_, rate)| units::line(OUTPUT_IMAGE, Quantity::Count(1), *rate, Side::Completion))
            .collect();
        Some((rates.first().map(|(at, _)| at.clone()), lines))
    };
    let billed = from_deployment.or_else(from_entry).or_else(from_keyed);
    if let Some((at, lines)) = billed {
        return priced(source, header, at, lines);
    }
    let Some((at, row)) = fallback.and_then(|at| Some((at.clone(), source.row(at)?))) else {
        return units::no_row(header);
    };
    let flat = listed(row, &OUTPUT_IMAGE).unwrap_or(UnitPrice::ZERO);
    let pixel = listed(row, &OUTPUT_PIXEL);
    let lines = images
        .iter()
        .map(|image| match image.pixels.zip(pixel) {
            Some((pixels, rate)) => megapixels(pixels, rate),
            None => units::line(OUTPUT_IMAGE, Quantity::Count(1), flat, Side::Completion),
        })
        .collect();
    priced(source, header, Some(at), lines)
}

/// `_flat_cost_per_image`: the pixel rate for each whole megapixel the image started, multiplied as main does
fn megapixels(pixels: u64, rate: UnitPrice) -> Line {
    let started = pixels.div_ceil(MEGAPIXEL);
    let amount = rate.get() * MEGAPIXEL as f64 * started as f64;
    Line {
        meter: OUTPUT_PIXEL,
        quantity: Quantity::Count(started * MEGAPIXEL),
        unit_price: rate,
        amount: Usd::new(amount).unwrap_or(Usd::ZERO),
        fallback_from: None,
        side: Side::Completion,
    }
}

fn submit<S: RowSource>(
    source: &S,
    header: &Header,
    seconds: Option<f64>,
    clip_resolution: Option<&str>,
    image_resolution: Option<&str>,
    images: u64,
) -> PriceOutcome {
    let Some((at, row)) = entry(source, header) else {
        return units::no_row(header);
    };
    let listed = |meter: &Meter| units::listed(row, meter, Channel::Online);
    if listed(&BILLABLE_UNIT).is_some() {
        return Unpriced::MissingRates {
            row: at,
            missing: Vec::new(),
        }
        .into();
    }
    let clip = seconds.zip(clip_resolution).map(|(seconds, resolution)| {
        Meter::Output(OutputUnit::VideoClip(VideoClip {
            length: format!("{}s", seconds.round_ties_even()).into(),
            resolution: resolution.trim().to_lowercase().into(),
        }))
    });
    let video = clip
        .into_iter()
        .chain([PER_VIDEO])
        .find_map(|meter| Some((listed(&meter)?, meter)))
        .map(|(rate, meter)| units::line(meter, Quantity::Count(1), rate, Side::Completion));
    let image = || {
        sized(image_resolution)
            .into_iter()
            .chain([OUTPUT_IMAGE])
            .find_map(|meter| Some((listed(&meter)?, meter)))
            .map(|(rate, meter)| {
                units::line(meter, Quantity::Count(images), rate, Side::Completion)
            })
    };
    match video.or_else(image) {
        Some(line) => priced(source, header, Some(at), vec![line]),
        None => Unpriced::MissingRates {
            row: at,
            missing: vec![PER_VIDEO, OUTPUT_IMAGE],
        }
        .into(),
    }
}

fn priced<S: RowSource>(
    source: &S,
    header: &Header,
    at: Option<RowRef>,
    lines: Vec<Line>,
) -> PriceOutcome {
    let receipt = units::receipt(source, header, None, Billed::of(lines, Vec::new()));
    Receipt {
        row: at,
        candidate: header.candidates.first().cloned(),
        ..receipt
    }
    .into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::testing::{BASE, TestRow, TestSource, header, prices, row_ref};

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

    fn query(keyed: Option<RowRef>, fallback: Option<RowRef>) -> FalQuery {
        FalQuery {
            header: header("fal_ai", "gpt-image-2"),
            call: FalCall::Image {
                images: vec![FalImage {
                    keyed,
                    pixels: Some(3 * MEGAPIXEL + 1),
                }],
                billable_units: None,
                resolution: None,
                fallback,
            },
        }
    }

    #[rstest]
    fn an_image_no_earlier_price_covers_asks_for_the_fallback_row() {
        let source = TestSource::with_row("gpt-image-2", row(&[]));
        assert!(matches!(
            price(&source, &query(None, None)),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }

    #[rstest]
    fn a_keyed_row_prices_the_image_before_the_fallback() {
        let source = TestSource::with_row("gpt-image-2", row(&[]))
            .and_row("keyed", row(&[(OUTPUT_IMAGE, 0.2)]))
            .and_row("fallback", row(&[(OUTPUT_IMAGE, 9.0)]));
        let outcome = price(
            &source,
            &query(Some(row_ref("keyed")), Some(row_ref("fallback"))),
        );
        assert!(matches!(outcome, PriceOutcome::Priced(receipt) if receipt.total == 0.2));
    }

    #[rstest]
    fn the_fallback_bills_every_megapixel_the_image_started() {
        let source = TestSource::with_row("gpt-image-2", row(&[])).and_row(
            "fallback",
            row(&[(OUTPUT_IMAGE, 9.0), (OUTPUT_PIXEL, 1e-8)]),
        );
        let outcome = price(&source, &query(None, Some(row_ref("fallback"))));
        assert!(matches!(
            outcome,
            PriceOutcome::Priced(receipt) if receipt.total == 1e-8 * MEGAPIXEL as f64 * 4.0
        ));
    }
}
