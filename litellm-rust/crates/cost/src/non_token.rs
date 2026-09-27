#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unit {
    Image,
    Pixel,
    Second,
    Page,
    Credit,
    Request,
    GuardrailUnit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Charge {
    pub unit: Unit,
    pub quantity: f64,
    pub rate: f64,
    pub units_per_rate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Component {
    pub unit: Unit,
    pub amount: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    pub components: Vec<Component>,
    pub total: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRates {
    pub input_per_image: Rate,
    pub output_per_image: Rate,
    pub input_per_pixel: Rate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageUsage {
    pub count: u64,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcrRates {
    pub per_credit: Rate,
    pub per_page: Rate,
    pub per_annotation_page: Rate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcrUsage {
    pub credits: Option<f64>,
    pub pages: u64,
    pub annotation_pages: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcrBatchRates {
    pub per_page_batch: Rate,
    pub per_page: Rate,
    pub per_annotation_page_batch: Rate,
    pub per_annotation_page: Rate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VideoRates<'a> {
    pub per_video_second: Rate,
    pub per_second: Rate,
    pub resolution_rates: &'a [(&'a str, Rate)],
}

fn priced(rate: Rate) -> Option<f64> {
    match rate {
        Rate::Value(value) => Some(value),
        Rate::Missing | Rate::Null | Rate::Invalid => None,
    }
}

pub fn calculate(charges: &[Charge]) -> Result<Cost, CostError> {
    let components: Vec<Component> = charges
        .iter()
        .map(|charge| {
            if !charge.quantity.is_finite() || charge.quantity < 0.0 {
                return Err(CostError::InvalidQuantity);
            }
            if !charge.rate.is_finite() || charge.rate < 0.0 {
                return Err(CostError::InvalidRate);
            }
            if !charge.units_per_rate.is_finite() || charge.units_per_rate <= 0.0 {
                return Err(CostError::InvalidRateBasis);
            }
            let amount = charge.quantity * charge.rate / charge.units_per_rate;
            if !amount.is_finite() {
                return Err(CostError::NonFiniteCost);
            }
            Ok(Component {
                unit: charge.unit,
                amount,
            })
        })
        .collect::<Result<_, _>>()?;
    let total: f64 = components.iter().map(|component| component.amount).sum();
    if !total.is_finite() {
        return Err(CostError::NonFiniteCost);
    }
    Ok(Cost { components, total })
}

pub fn calculate_image(tables: &[ImageRates], usage: ImageUsage) -> Result<Cost, CostError> {
    let selected = tables
        .iter()
        .find_map(|rates| {
            priced(rates.input_per_image)
                .map(|rate| (Unit::Image, usage.count as f64, rate))
                .or_else(|| {
                    priced(rates.output_per_image)
                        .map(|rate| (Unit::Image, usage.count as f64, rate))
                })
                .or_else(|| {
                    priced(rates.input_per_pixel).map(|rate| {
                        (
                            Unit::Pixel,
                            usage.count as f64 * usage.width as f64 * usage.height as f64,
                            rate,
                        )
                    })
                })
        })
        .ok_or(CostError::MissingRate)?;
    calculate(&[Charge {
        unit: selected.0,
        quantity: selected.1,
        rate: selected.2,
        units_per_rate: 1.0,
    }])
}

pub fn calculate_ocr(rates: OcrRates, usage: OcrUsage) -> Result<Cost, CostError> {
    if let (Some(credits), Some(rate)) = (usage.credits, priced(rates.per_credit)) {
        return calculate(&[Charge {
            unit: Unit::Credit,
            quantity: credits,
            rate,
            units_per_rate: 1.0,
        }]);
    }
    let page_rate = priced(rates.per_page);
    let annotation_rate = priced(rates.per_annotation_page).or(page_rate);
    if page_rate.is_none() && (annotation_rate.is_none() || usage.annotation_pages == 0) {
        return calculate(&[]);
    }
    let charges: Vec<Charge> = [
        page_rate.map(|rate| Charge {
            unit: Unit::Page,
            quantity: usage.pages as f64,
            rate,
            units_per_rate: 1.0,
        }),
        annotation_rate.map(|rate| Charge {
            unit: Unit::Page,
            quantity: usage.annotation_pages as f64,
            rate,
            units_per_rate: 1.0,
        }),
    ]
    .into_iter()
    .flatten()
    .collect();
    calculate(&charges)
}

pub fn calculate_ocr_with_tables(tables: &[OcrRates], usage: OcrUsage) -> Result<Cost, CostError> {
    let rates = OcrRates {
        per_credit: tables
            .iter()
            .find_map(|table| priced(table.per_credit))
            .map_or(Rate::Missing, Rate::Value),
        per_page: tables
            .iter()
            .find_map(|table| priced(table.per_page))
            .map_or(Rate::Missing, Rate::Value),
        per_annotation_page: tables
            .iter()
            .find_map(|table| priced(table.per_annotation_page))
            .map_or(Rate::Missing, Rate::Value),
    };
    calculate_ocr(rates, usage)
}

pub fn calculate_ocr_batch(
    deployment: Option<OcrBatchRates>,
    published: Option<OcrBatchRates>,
    pages: u64,
    annotation_pages: u64,
) -> Result<Cost, CostError> {
    let family_rate = |select: fn(OcrBatchRates) -> (Rate, Rate)| {
        [deployment, published]
            .into_iter()
            .flatten()
            .find_map(|rates| {
                let (batch, standard) = select(rates);
                priced(batch).or_else(|| priced(standard))
            })
    };
    let page_rate = family_rate(|rates| (rates.per_page_batch, rates.per_page));
    let annotation_rate =
        family_rate(|rates| (rates.per_annotation_page_batch, rates.per_annotation_page))
            .or(page_rate);
    let charges: Vec<_> = [
        page_rate.map(|rate| Charge {
            unit: Unit::Page,
            quantity: pages as f64,
            rate,
            units_per_rate: 1.0,
        }),
        annotation_rate.map(|rate| Charge {
            unit: Unit::Page,
            quantity: annotation_pages as f64,
            rate,
            units_per_rate: 1.0,
        }),
    ]
    .into_iter()
    .flatten()
    .collect();
    calculate(&charges)
}

pub fn calculate_video(
    rates: &VideoRates<'_>,
    duration_seconds: f64,
    resolution: Option<&str>,
) -> Result<Cost, CostError> {
    let resolution_rate = resolution
        .and_then(video_resolution_to_cost_field_suffix)
        .and_then(|suffix| {
            rates
                .resolution_rates
                .iter()
                .find(|(name, _)| *name == suffix)
                .and_then(|(_, rate)| priced(*rate))
        });
    let rate = priced(rates.per_video_second)
        .or(resolution_rate)
        .or_else(|| priced(rates.per_second));
    match rate {
        Some(rate) => calculate(&[Charge {
            unit: Unit::Second,
            quantity: duration_seconds,
            rate,
            units_per_rate: 1.0,
        }]),
        None => {
            if !duration_seconds.is_finite() || duration_seconds < 0.0 {
                return Err(CostError::InvalidQuantity);
            }
            calculate(&[])
        }
    }
}

pub fn video_resolution_to_cost_field_suffix(resolution: &str) -> Option<String> {
    let suffix: String = resolution
        .trim()
        .to_lowercase()
        .chars()
        .filter(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!suffix.is_empty() && suffix.chars().count() <= 24).then_some(suffix)
}
use crate::error::CostError;
use crate::pricing::Rate;
