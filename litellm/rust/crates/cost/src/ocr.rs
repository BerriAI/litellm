//! The `ocr` kind, a port of `ocr_cost` and `ocr_batch_cost` (`cost_calculator.py`). The caller passes the
//! deployment's price tables as `PriceSource::Rows`, read ahead of the catalog row. An online call reads each rate
//! from the first table that sets it and bills credits when it reported some and a credit rate exists, otherwise
//! pages and annotation pages, an annotation page falling back to the page rate. A batch line reads each family, the
//! `_batches` rate then the online one, from the first table that prices that family

use crate::money::UnitPrice;
use crate::price::{Channel, Feature, Meter};
use crate::priced::{Line, Quantity, Side};
use crate::query::Header;
use crate::receipt::{PriceOutcome, Unpriced};
use crate::row::{RowRef, RowSource};
use crate::units::{self, Billed};

const PAGE: Meter = Meter::Feature(Feature::OcrPage);
const ANNOTATION: Meter = Meter::Feature(Feature::AnnotationPage);
const CREDIT: Meter = Meter::Feature(Feature::OcrCredit);

/// One OCR result and the `usage_info` it reported
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct OcrQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// `pages_processed`, `None` when the response did not report it
    #[cfg_attr(feature = "serde", serde(default))]
    pub pages: Option<u64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub annotation_pages: u64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub credits: Option<f64>,
    /// A line of a batch output file, billed at the `_batches` rates
    #[cfg_attr(feature = "serde", serde(default))]
    pub batch: bool,
    /// The caller has not fetched the catalog row yet, as `ocr_batch_cost` reads it only for a family with pages
    /// that no table it was given prices
    #[cfg_attr(feature = "serde", serde(default))]
    pub catalog_deferred: bool,
}

/// A rate and the table it came from
type Found = (RowRef, UnitPrice);

pub(crate) fn price<S: RowSource>(source: &S, query: &OcrQuery) -> PriceOutcome {
    let layers = units::layers(source, &query.header);
    let read = |channels: &[Channel], meter: &Meter| -> Option<Found> {
        layers.iter().find_map(|(at, row)| {
            channels
                .iter()
                .find_map(|channel| units::listed(*row, meter, *channel))
                .map(|rate| (at.clone(), rate))
        })
    };
    let channels: &[Channel] = match query.batch {
        true => &[Channel::Batch, Channel::Online],
        false => &[Channel::Online],
    };
    let page = read(channels, &PAGE);
    let annotation = read(channels, &ANNOTATION);
    let unpriced_pages = |rate: &Option<Found>, pages: u64| pages > 0 && rate.is_none();
    if query.catalog_deferred
        && (unpriced_pages(&page, query.pages.unwrap_or(0))
            || unpriced_pages(&annotation, query.annotation_pages))
    {
        return Unpriced::CatalogDeferred.into();
    }
    let Some((first, _)) = layers.first() else {
        return units::no_row(&query.header);
    };
    let first = first.clone();
    let credit = read(&[Channel::Online], &CREDIT).filter(|_| !query.batch);
    if let (Some(credits), Some((at, rate))) = (query.credits, &credit) {
        let line = units::line(CREDIT, Quantity::Measured(credits), *rate, Side::Prompt);
        return priced(source, query, at, vec![line], Vec::new());
    }
    let annotation_rate = annotation.as_ref().or(page.as_ref());
    let bills_annotation = annotation_rate.is_some() && query.annotation_pages > 0;
    if !query.batch && !bills_annotation {
        match (query.pages, &page) {
            (None, Some((at, _))) if credit.is_none() => {
                return Unpriced::MissingUsage {
                    row: at.clone(),
                    meter: PAGE,
                }
                .into();
            }
            (None, _) | (Some(_), None) => {
                return Unpriced::MissingRates {
                    row: first,
                    missing: vec![PAGE],
                }
                .into();
            }
            (Some(_), Some(_)) => {}
        }
    }
    let annotation_line = annotation_rate.map(|(_, rate)| {
        let line = units::line(
            ANNOTATION,
            Quantity::Count(query.annotation_pages),
            *rate,
            Side::Prompt,
        );
        Line {
            fallback_from: annotation.is_none().then_some(PAGE),
            ..line
        }
    });
    let charges = [
        (
            PAGE,
            query.pages.unwrap_or(0),
            page.as_ref().map(|(_, rate)| {
                units::line(
                    PAGE,
                    Quantity::Count(query.pages.unwrap_or(0)),
                    *rate,
                    Side::Prompt,
                )
            }),
        ),
        (ANNOTATION, query.annotation_pages, annotation_line),
    ];
    let used = charges.into_iter().filter(|(_, quantity, _)| *quantity > 0);
    let (lines, missing): (Vec<_>, Vec<_>) = used.partition(|(_, _, line)| line.is_some());
    let lines: Vec<Line> = lines.into_iter().filter_map(|(_, _, line)| line).collect();
    let missing: Vec<Meter> = missing.into_iter().map(|(meter, _, _)| meter).collect();
    if lines.is_empty() && !missing.is_empty() {
        return Unpriced::MissingRates {
            row: first,
            missing,
        }
        .into();
    }
    let at = page.or(annotation).map_or(first, |(at, _)| at);
    priced(source, query, &at, lines, missing)
}

fn priced<S: RowSource>(
    source: &S,
    query: &OcrQuery,
    at: &RowRef,
    lines: Vec<Line>,
    missing: Vec<Meter>,
) -> PriceOutcome {
    let receipt = units::receipt(source, &query.header, None, Billed::of(lines, missing));
    crate::receipt::Receipt {
        row: Some(at.clone()),
        candidate: query.header.candidates.first().cloned(),
        ..receipt
    }
    .into()
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::{Band, Price, Prices, When};
    use crate::query::PriceSource;
    use crate::testing::{TestRow, TestSource, header, priced as receipt_of, row_ref};

    fn row(rates: &[(Meter, Channel, f64)]) -> TestRow {
        TestRow {
            prices: Prices::new(
                rates
                    .iter()
                    .map(|(meter, channel, rate)| {
                        (
                            Price {
                                meter: meter.clone(),
                                channel: *channel,
                                when: When {
                                    band: Band::Base,
                                    ..When::default()
                                },
                            },
                            UnitPrice::new(*rate).unwrap(),
                        )
                    })
                    .collect(),
                None,
                Default::default(),
            ),
            ..TestRow::default()
        }
    }

    fn query(
        pages: Option<u64>,
        annotation_pages: u64,
        credits: Option<f64>,
        batch: bool,
    ) -> OcrQuery {
        OcrQuery {
            header: header("mistral", "mistral-ocr"),
            pages,
            annotation_pages,
            credits,
            batch,
            catalog_deferred: false,
        }
    }

    fn with_deployment(query: OcrQuery) -> OcrQuery {
        OcrQuery {
            header: Header {
                source: PriceSource::Rows {
                    rows: vec![row_ref("deployment")],
                },
                ..query.header
            },
            ..query
        }
    }

    fn catalog() -> TestSource {
        TestSource::with_row(
            "mistral-ocr",
            row(&[
                (PAGE, Channel::Online, 0.001),
                (ANNOTATION, Channel::Online, 0.003),
                (PAGE, Channel::Batch, 0.0005),
            ]),
        )
    }

    fn total(outcome: PriceOutcome) -> f64 {
        receipt_of(outcome).total
    }

    #[rstest]
    #[case::pages(Some(10), 0, 0.001 * 10.0)]
    #[case::pages_and_annotations(Some(10), 4, 0.001 * 10.0 + 0.003 * 4.0)]
    #[case::annotations_without_pages(None, 4, 0.003 * 4.0)]
    fn an_online_call_bills_pages_and_annotation_pages(
        #[case] pages: Option<u64>,
        #[case] annotation_pages: u64,
        #[case] expected: f64,
    ) {
        assert_eq!(
            total(price(
                &catalog(),
                &query(pages, annotation_pages, None, false)
            )),
            expected
        );
    }

    #[rstest]
    fn an_annotation_page_without_its_own_rate_bills_at_the_page_rate() {
        let source = TestSource::with_row("mistral-ocr", row(&[(PAGE, Channel::Online, 0.002)]));
        let receipt = receipt_of(price(&source, &query(Some(1), 3, None, false)));
        assert_eq!(receipt.total, 0.002 + 0.002 * 3.0);
        assert_eq!(receipt.lines[1].fallback_from, Some(PAGE));
    }

    #[rstest]
    fn reported_credits_bill_at_the_credit_rate_instead_of_pages() {
        let source = TestSource::with_row(
            "mistral-ocr",
            row(&[
                (CREDIT, Channel::Online, 0.01),
                (PAGE, Channel::Online, 1.0),
            ]),
        );
        assert_eq!(
            total(price(&source, &query(Some(5), 0, Some(2.5), false))),
            0.01 * 2.5
        );
    }

    #[rstest]
    fn a_deployment_rate_wins_field_by_field_over_the_catalog() {
        let source = catalog().and_row("deployment", row(&[(PAGE, Channel::Online, 0.5)]));
        let receipt = receipt_of(price(
            &source,
            &with_deployment(query(Some(2), 1, None, false)),
        ));
        assert_eq!(receipt.total, 0.5 * 2.0 + 0.003);
    }

    #[rstest]
    fn missing_pages_on_a_page_priced_row_are_missing_usage() {
        assert!(matches!(
            price(&catalog(), &query(None, 0, None, false)),
            PriceOutcome::Unpriced(Unpriced::MissingUsage { .. })
        ));
    }

    #[rstest]
    #[case::credit_priced_without_credits(
        row(&[(CREDIT, Channel::Online, 0.01), (PAGE, Channel::Online, 1.0)]),
        query(None, 0, None, false)
    )]
    #[case::no_page_rate(row(&[(CREDIT, Channel::Online, 0.01)]), query(Some(3), 0, None, false))]
    fn an_online_call_with_nothing_billable_is_missing_its_page_rate(
        #[case] priced_row: TestRow,
        #[case] call: OcrQuery,
    ) {
        assert!(matches!(
            price(&TestSource::with_row("mistral-ocr", priced_row), &call),
            PriceOutcome::Unpriced(Unpriced::MissingRates { .. })
        ));
    }

    #[rstest]
    #[case::batch_rate(catalog(), 0.0005 * 4.0 + 0.003 * 2.0)]
    #[case::online_rate_without_a_batch_one(
        TestSource::with_row("mistral-ocr", row(&[(PAGE, Channel::Online, 0.001)])),
        0.001 * 4.0 + 0.001 * 2.0
    )]
    fn a_batch_line_reads_the_batch_rate_then_the_online_one(
        #[case] source: TestSource,
        #[case] expected: f64,
    ) {
        assert_eq!(
            total(price(&source, &query(Some(4), 2, None, true))),
            expected
        );
    }

    #[rstest]
    #[case::page_family_unpriced(row(&[(ANNOTATION, Channel::Batch, 0.1)]), true)]
    #[case::annotation_family_unpriced(row(&[(PAGE, Channel::Batch, 0.1)]), true)]
    #[case::both_families_priced(
        row(&[(PAGE, Channel::Online, 0.1), (ANNOTATION, Channel::Batch, 0.2)]),
        false
    )]
    fn a_deferred_catalog_is_asked_for_only_when_a_family_with_pages_is_unpriced(
        #[case] deployment: TestRow,
        #[case] asks: bool,
    ) {
        let source = TestSource::default().and_row("deployment", deployment);
        let deferred = OcrQuery {
            catalog_deferred: true,
            ..with_deployment(query(Some(4), 2, None, true))
        };
        assert_eq!(
            price(&source, &deferred) == Unpriced::CatalogDeferred.into(),
            asks
        );
    }

    #[rstest]
    fn a_batch_family_the_deployment_prices_ignores_the_catalog() {
        let source = catalog().and_row("deployment", row(&[(PAGE, Channel::Online, 0.25)]));
        assert_eq!(
            total(price(
                &source,
                &with_deployment(query(Some(4), 2, None, true))
            )),
            0.25 * 4.0 + 0.003 * 2.0
        );
    }
}
