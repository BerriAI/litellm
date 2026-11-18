//! What the kinds billed per unit share: the rows a query names, the call those kinds bill at, one row billed through
//! `calculate`, and a receipt over lines with the query's discount and margin settled. Which meter and quantity a
//! call bills is each kind's own rule, in its own module

use crate::adjust::{Discount, Margin};
use crate::calculate;
use crate::call::CallContext;
use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, Meter, Price, ServiceTier, When};
use crate::priced::{Line, Outcome, Quantity, Side};
use crate::query::{Basis, Candidate, Header, PriceSource};
use crate::receipt::{Breakdown, PriceOutcome, Receipt, Unpriced};
use crate::row::{LoweredRow, RowRef, RowSource};
use crate::usage::Usage;

/// A row a query prices with, and the candidate it was resolved under
pub(crate) struct Named<'a, R> {
    pub candidate: &'a Candidate,
    pub at: RowRef,
    pub row: &'a R,
}

/// The row the header names: the given row under the first candidate, or the first candidate the catalog resolves
pub(crate) fn named<'a, S: RowSource>(
    source: &'a S,
    header: &'a Header,
) -> Option<Named<'a, S::Row>> {
    match &header.source {
        PriceSource::Row { row: at, .. } => Some(Named {
            candidate: header.candidates.first()?,
            row: source.row(at)?,
            at: at.clone(),
        }),
        PriceSource::Catalog => header.candidates.iter().find_map(|candidate| {
            let at = source.resolve(candidate)?;
            Some(Named {
                candidate,
                row: source.row(&at)?,
                at,
            })
        }),
        PriceSource::Rows { .. }
        | PriceSource::FlatPerToken(_)
        | PriceSource::PerSecond { .. }
        | PriceSource::PerQuery { .. } => None,
    }
}

/// The rows the caller passed, then the first candidate's catalog row, in the order main reads its price tables
pub(crate) fn layers<'a, S: RowSource>(
    source: &'a S,
    header: &'a Header,
) -> Vec<(RowRef, &'a S::Row)> {
    let given: &[RowRef] = match &header.source {
        PriceSource::Rows { rows } => rows,
        _ => &[],
    };
    let catalog = header
        .candidates
        .iter()
        .find_map(|candidate| source.resolve(candidate));
    given
        .iter()
        .cloned()
        .chain(catalog)
        .filter_map(|at| Some((at.clone(), source.row(&at)?)))
        .collect()
}

pub(crate) fn no_row(header: &Header) -> PriceOutcome {
    Unpriced::NoRow {
        tried: header.candidates.clone(),
    }
    .into()
}

/// An online call at the standard tier with no uplift, as main bills every per-unit kind
pub(crate) fn call(basis: &Basis) -> CallContext {
    CallContext {
        tier: ServiceTier::Standard,
        channel: Channel::Online,
        residency: None,
        fedramp_endpoint: false,
        non_global_endpoint: false,
        speed: None,
        inference_geo: None,
        at: basis.at,
        overlay: None,
    }
}

/// The row's own rate for `meter` on `channel`, with no fallback
pub(crate) fn listed(row: &impl LoweredRow, meter: &Meter, channel: Channel) -> Option<UnitPrice> {
    row.prices().flat().get(&Price {
        meter: meter.clone(),
        channel,
        when: When::default(),
    })
}

/// Bills `usage` on one row through `calculate`. A row that prices none of the meters the call used is unpriced
pub(crate) fn bill<S: RowSource>(
    source: &S,
    header: &Header,
    named: &Named<'_, S::Row>,
    usage: &Usage,
    call: &CallContext,
) -> PriceOutcome {
    let prices = named.row.prices().for_prompt(usage.prompt_tokens);
    match calculate(prices, named.row.adjustments(), usage, call) {
        Outcome::Unpriced { missing } => Unpriced::MissingRates {
            row: named.at.clone(),
            missing,
        }
        .into(),
        Outcome::Priced(priced) => receipt(
            source,
            header,
            Some(named),
            Billed {
                lines: priced.lines,
                missing: priced.missing,
                prompt: priced.prompt,
                completion: priced.completion,
            },
        )
        .into(),
    }
}

/// Billed lines, the meters the call used that no row prices, and each side's total
pub(crate) struct Billed {
    pub lines: Vec<Line>,
    pub missing: Vec<Meter>,
    pub prompt: Usd,
    pub completion: Usd,
}

impl Billed {
    /// Lines a kind billed itself, each side summed in line order
    pub(crate) fn of(lines: Vec<Line>, missing: Vec<Meter>) -> Self {
        let side = |side: Side| {
            lines
                .iter()
                .filter(|line| line.side == side)
                .map(|line| line.amount)
                .sum::<Usd>()
        };
        Self {
            prompt: side(Side::Prompt),
            completion: side(Side::Completion),
            lines,
            missing,
        }
    }
}

pub(crate) fn line(meter: Meter, quantity: Quantity, unit_price: UnitPrice, side: Side) -> Line {
    Line {
        meter,
        quantity,
        unit_price,
        amount: unit_price * quantity,
        fallback_from: None,
        side,
    }
}

/// The receipt of `billed` with the header's discount and margin settled
pub(crate) fn receipt<S: RowSource>(
    source: &S,
    header: &Header,
    named: Option<&Named<'_, S::Row>>,
    billed: Billed,
) -> Receipt {
    let model = named.map(|named| &*named.candidate.model);
    Receipt {
        candidate: named.map(|named| named.candidate.clone()),
        row: named.map(|named| named.at.clone()),
        prompt: billed.prompt,
        completion: billed.completion,
        lines: billed.lines,
        defaulted_to_zero: billed.missing,
        rates: None,
        tools: Usd::ZERO,
        additional: Usd::ZERO,
        discount: Discount::default(),
        margin: Margin::default(),
        breakdown: Breakdown::default(),
        parts: Vec::new(),
        cache_savings: None,
        reported: None,
        basis: header.basis.clone(),
        total: 0.0,
    }
    .settled(source, &header.adjustments, model)
}
