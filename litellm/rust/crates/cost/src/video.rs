//! The `video` kind, a port of the video branch of `completion_cost` (`cost_calculator.py`), `video_generation_cost`
//! and `video_clip_cost` (`llms/openai/cost_calculation.py`) and `default_video_cost_calculator`. One video bills the
//! first of: the clip table cell for its rounded length and resolution, `output_cost_per_video`,
//! `output_cost_per_video_per_second`, `output_cost_per_second_<resolution>` or `output_cost_per_second`, each read
//! off the deployment's own table when it has one and the catalog row otherwise. That cost times the videos, plus the
//! billable input images at the first of the `input_image_rows` that prices an input image. Discount and margin never
//! apply

use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, InputUnit, Meter, OutputUnit, Resolution, VideoClip};
use crate::priced::{Line, Quantity, Side};
use crate::query::Header;
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::{LoweredRow, RowRef, RowSource};
use crate::units::{self, Billed};

const PER_VIDEO: Meter = Meter::Output(OutputUnit::Video);
const PER_VIDEO_SECOND: Meter = Meter::Output(OutputUnit::VideoSecond);
const INPUT_IMAGE: Meter = Meter::Input(InputUnit::Image);

/// One video generation and what its usage reported
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct VideoQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// `duration_seconds`, `None` when the usage reported none, which bills a zero length
    #[cfg_attr(feature = "serde", serde(default))]
    pub seconds: Option<f64>,
    /// `video_resolution`, lowercased
    #[cfg_attr(feature = "serde", serde(default))]
    pub resolution: Option<Box<str>>,
    pub videos: u64,
    /// `billable_input_image_count`
    #[cfg_attr(feature = "serde", serde(default))]
    pub input_images: u64,
    /// The tables an input image rate is read from, in order
    #[cfg_attr(feature = "serde", serde(default))]
    pub input_image_rows: Vec<RowRef>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &VideoQuery) -> PriceOutcome {
    let layers = units::layers(source, &query.header);
    let Some((at, row)) = layers.first() else {
        return units::no_row(&query.header);
    };
    let seconds = query.seconds.unwrap_or(0.0);
    let video = generated(*row, query, seconds).map(|(meter, rate, quantity)| {
        let one = rate * quantity;
        Line {
            meter,
            quantity,
            unit_price: rate,
            amount: Usd::new(one.get() * query.videos as f64).unwrap_or(one),
            fallback_from: None,
            side: Side::Completion,
        }
    });
    let image_rate = query
        .input_image_rows
        .iter()
        .filter_map(|at| source.row(at))
        .find_map(|row| units::listed(row, &INPUT_IMAGE, Channel::Online));
    let image = image_rate.filter(|_| query.input_images > 0).map(|rate| {
        units::line(
            INPUT_IMAGE,
            Quantity::Count(query.input_images),
            rate,
            Side::Prompt,
        )
    });
    let missing: Vec<Meter> = video.is_none().then_some(PER_VIDEO).into_iter().collect();
    let lines: Vec<Line> = video.into_iter().chain(image).collect();
    if lines.is_empty() {
        return Unpriced::MissingRates {
            row: at.clone(),
            missing,
        }
        .into();
    }
    let receipt = units::receipt(source, &query.header, None, Billed::of(lines, missing));
    Receipt {
        row: Some(at.clone()),
        candidate: query.header.candidates.first().cloned(),
        ..receipt
    }
    .into()
}

/// The meter one video bills at, its rate and quantity, in main's order
fn generated(
    row: &impl LoweredRow,
    query: &VideoQuery,
    seconds: f64,
) -> Option<(Meter, UnitPrice, Quantity)> {
    let listed = |meter: &Meter| units::listed(row, meter, Channel::Online);
    let clip = query
        .resolution
        .as_deref()
        .filter(|_| seconds > 0.0)
        .map(|resolution| {
            Meter::Output(OutputUnit::VideoClip(VideoClip {
                length: format!("{}s", seconds.round_ties_even()).into(),
                resolution: resolution.trim().to_lowercase().into(),
            }))
        });
    let at_resolution = Meter::Output(OutputUnit::Second(
        query.resolution.as_deref().and_then(Resolution::new),
    ));
    let per_second = || {
        listed(&at_resolution)
            .or_else(|| listed(&Meter::Output(OutputUnit::Second(None))))
            .map(|rate| (at_resolution.clone(), rate, Quantity::Measured(seconds)))
    };
    clip.and_then(|clip| Some((clip.clone(), listed(&clip)?, Quantity::Count(1))))
        .or_else(|| Some((PER_VIDEO, listed(&PER_VIDEO)?, Quantity::Count(1))))
        .or_else(|| {
            Some((
                PER_VIDEO_SECOND,
                listed(&PER_VIDEO_SECOND)?,
                Quantity::Measured(seconds),
            ))
        })
        .or_else(per_second)
}

#[cfg(test)]
mod parity {
    //! Parity with main's `video_generation_cost`. `fixtures/video_parity.json` was captured once from main's Python: each
    //! case holds the cost fields `get_model_info` returned, the reported length and resolution, and main's total. The
    //! adapter answers a video no rate prices with 0.0, as main did

    use rstest::rstest;

    use super::{VideoQuery, price};
    use crate::receipt::PriceOutcome;
    use crate::testing::{TestSource, fixture_case, header, units_row};

    const FIXTURE: &str = include_str!("../fixtures/video_parity.json");

    #[rstest]
    #[case::sora_per_second("sora_per_second")]
    #[case::sora_pro_resolution_rate("sora_pro_resolution_rate")]
    #[case::clip_cell("clip_cell")]
    #[case::clip_cell_rounds_half_to_even("clip_cell_rounds_half_to_even")]
    #[case::clip_without_cell_is_unpriced("clip_without_cell_is_unpriced")]
    #[case::resolution_rate("resolution_rate")]
    #[case::resolution_without_rate("resolution_without_rate")]
    #[case::per_video("per_video")]
    #[case::measured_length("measured_length")]
    fn a_video_costs_what_main_charged(#[case] name: &str) {
        let case = fixture_case(FIXTURE, name);
        let model = case["model"].as_str().unwrap();
        let source = TestSource::with_row(model, units_row(&case["row"]));
        let query = VideoQuery {
            header: header("openai", model),
            seconds: case["seconds"].as_f64(),
            resolution: case["resolution"]
                .as_str()
                .map(|resolution| resolution.to_lowercase().into()),
            videos: 1,
            input_images: 0,
            input_image_rows: Vec::new(),
        };
        let total = match price(&source, &query) {
            PriceOutcome::Priced(receipt) => receipt.total,
            PriceOutcome::Unpriced(_) => 0.0,
        };
        assert_eq!(Some(total), case["expected"].as_f64());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::query::PriceSource;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref};

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

    fn clip(length: &str, resolution: &str) -> Meter {
        Meter::Output(OutputUnit::VideoClip(VideoClip {
            length: length.into(),
            resolution: resolution.into(),
        }))
    }

    fn second(resolution: Option<&str>) -> Meter {
        Meter::Output(OutputUnit::Second(resolution.and_then(Resolution::new)))
    }

    fn query(seconds: Option<f64>, resolution: Option<&str>, videos: u64) -> VideoQuery {
        VideoQuery {
            header: header("openai", "sora-2"),
            seconds,
            resolution: resolution.map(Into::into),
            videos,
            input_images: 0,
            input_image_rows: Vec::new(),
        }
    }

    fn total(source: &TestSource, video: &VideoQuery) -> f64 {
        priced(price(source, video)).total
    }

    fn every_rate() -> TestRow {
        row(&[
            (clip("8s", "720p"), 0.8),
            (PER_VIDEO, 0.5),
            (PER_VIDEO_SECOND, 0.2),
            (second(Some("720p")), 0.15),
            (second(None), 0.1),
        ])
    }

    #[rstest]
    #[case::clip_cell_rounds_half_to_even(Some(8.5), Some("720P"), 0.8)]
    #[case::no_cell_for_the_length(Some(4.0), Some("720p"), 0.5)]
    #[case::no_length(None, Some("720p"), 0.5)]
    fn a_clip_cell_wins_then_the_per_video_rate(
        #[case] seconds: Option<f64>,
        #[case] resolution: Option<&str>,
        #[case] expected: f64,
    ) {
        let source = TestSource::with_row("sora-2", every_rate());
        assert_eq!(total(&source, &query(seconds, resolution, 1)), expected);
    }

    #[rstest]
    #[case::per_video_second(row(&[(PER_VIDEO_SECOND, 0.2), (second(None), 0.1)]), 0.2 * 6.5)]
    #[case::resolution_rate(row(&[(second(Some("720p")), 0.15), (second(None), 0.1)]), 0.15 * 6.5)]
    #[case::plain_second_rate(row(&[(second(None), 0.1)]), 0.1 * 6.5)]
    fn without_a_per_video_rate_the_length_bills_by_the_second(
        #[case] video_row: TestRow,
        #[case] expected: f64,
    ) {
        let source = TestSource::with_row("sora-2", video_row);
        assert_eq!(total(&source, &query(Some(6.5), Some("720p"), 1)), expected);
    }

    #[rstest]
    fn each_video_costs_the_same() {
        let source = TestSource::with_row("sora-2", row(&[(second(None), 0.1)]));
        assert_eq!(total(&source, &query(Some(6.5), None, 3)), 0.1 * 6.5 * 3.0);
    }

    #[rstest]
    fn a_deployment_table_replaces_the_catalog_row() {
        let source = TestSource::with_row("sora-2", every_rate())
            .and_row("deployment", row(&[(second(None), 0.01)]));
        let deployed = VideoQuery {
            header: Header {
                source: PriceSource::Rows {
                    rows: vec![row_ref("deployment")],
                },
                ..header("openai", "sora-2")
            },
            ..query(Some(4.0), Some("720p"), 1)
        };
        assert_eq!(total(&source, &deployed), 0.01 * 4.0);
    }

    #[rstest]
    fn billable_input_images_add_the_first_rows_image_rate() {
        let source = TestSource::with_row("sora-2", row(&[(PER_VIDEO, 0.5)]))
            .and_row("unpriced", row(&[]))
            .and_row("requested", row(&[(INPUT_IMAGE, 0.02)]));
        let with_images = VideoQuery {
            input_images: 3,
            input_image_rows: vec![row_ref("unpriced"), row_ref("requested")],
            ..query(Some(4.0), None, 2)
        };
        let receipt = priced(price(&source, &with_images));
        assert_eq!(receipt.total, 0.5 * 2.0 + 0.02 * 3.0);
    }

    #[rstest]
    fn a_row_with_no_video_rate_is_missing_it() {
        let source = TestSource::with_row("sora-2", row(&[]));
        assert!(matches!(
            price(&source, &query(Some(4.0), None, 1)),
            PriceOutcome::Unpriced(Unpriced::MissingRates { .. })
        ));
    }
}
