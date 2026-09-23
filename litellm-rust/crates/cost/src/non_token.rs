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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    MissingRate,
    InvalidQuantity,
    InvalidRate,
    InvalidRateBasis,
    NonFiniteCost,
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

fn priced(rate: Rate) -> Option<f64> {
    match rate {
        Rate::Value(value) => Some(value),
        Rate::Missing | Rate::Null => None,
    }
}

pub fn calculate(charges: &[Charge]) -> Result<Cost, Error> {
    let components: Vec<Component> = charges
        .iter()
        .map(|charge| {
            if !charge.quantity.is_finite() || charge.quantity < 0.0 {
                return Err(Error::InvalidQuantity);
            }
            if !charge.rate.is_finite() || charge.rate < 0.0 {
                return Err(Error::InvalidRate);
            }
            if !charge.units_per_rate.is_finite() || charge.units_per_rate <= 0.0 {
                return Err(Error::InvalidRateBasis);
            }
            let amount = charge.quantity * charge.rate / charge.units_per_rate;
            if !amount.is_finite() {
                return Err(Error::NonFiniteCost);
            }
            Ok(Component {
                unit: charge.unit,
                amount,
            })
        })
        .collect::<Result<_, _>>()?;
    let total: f64 = components.iter().map(|component| component.amount).sum();
    if !total.is_finite() {
        return Err(Error::NonFiniteCost);
    }
    Ok(Cost { components, total })
}

pub fn calculate_image(tables: &[ImageRates], usage: ImageUsage) -> Result<Cost, Error> {
    if usage.width == 0 || usage.height == 0 {
        return Err(Error::InvalidQuantity);
    }
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
        .ok_or(Error::MissingRate)?;
    calculate(&[Charge {
        unit: selected.0,
        quantity: selected.1,
        rate: selected.2,
        units_per_rate: 1.0,
    }])
}

pub fn calculate_ocr(rates: OcrRates, usage: OcrUsage) -> Result<Cost, Error> {
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

pub fn calculate_ocr_with_tables(tables: &[OcrRates], usage: OcrUsage) -> Result<Cost, Error> {
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
use crate::Rate;
