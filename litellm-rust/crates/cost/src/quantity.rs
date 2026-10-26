use crate::PriceEvaluationError;
use crate::price::{Feature, InputUnit, Meter, OutputUnit};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quantity {
    Count(u64),
    Measured(Measurement),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement(f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::VariantArray)]
pub enum QuantityUnit {
    #[strum(serialize = "meter")]
    Meter,
    #[strum(serialize = "milliseconds")]
    Milliseconds,
}

impl Measurement {
    pub fn new(value: f64) -> Result<Self, PriceEvaluationError> {
        if !value.is_finite() || value < 0.0 {
            return Err(PriceEvaluationError::InvalidQuantity);
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl Quantity {
    pub fn measured(
        value: Measurement,
        unit: QuantityUnit,
        meter: &Meter,
    ) -> Result<Self, PriceEvaluationError> {
        let seconds = matches!(
            meter,
            Meter::WallClockSecond
                | Meter::Input(
                    InputUnit::Second | InputUnit::AudioSecond | InputUnit::VideoSecond { .. }
                )
                | Meter::Output(OutputUnit::Second(_) | OutputUnit::VideoSecond)
        );
        let characters = matches!(
            meter,
            Meter::Input(InputUnit::Character) | Meter::Output(OutputUnit::Character)
        );
        let storage = matches!(
            meter,
            Meter::Feature(Feature::FileSearchGbDay | Feature::VectorStoreGbDay)
        );
        match unit {
            QuantityUnit::Meter if seconds || characters || storage => Ok(Self::Measured(value)),
            QuantityUnit::Milliseconds if seconds => {
                Measurement::new(value.get() / 1000.0).map(Self::Measured)
            }
            _ => Err(PriceEvaluationError::InvalidQuantityUnit),
        }
    }

    pub fn is_zero(self) -> bool {
        match self {
            Self::Count(value) => value == 0,
            Self::Measured(value) => value.get() == 0.0,
        }
    }

    pub fn exceeds(self, threshold: u64) -> bool {
        match self {
            Self::Count(value) => value > threshold,
            Self::Measured(value) => value.get() > threshold as f64,
        }
    }

    pub fn exceeds_scaled(self, threshold: u64, scale: u64) -> Result<bool, PriceEvaluationError> {
        match self {
            Self::Count(value) => Ok(u128::from(value) * u128::from(scale) > u128::from(threshold)),
            Self::Measured(value) => {
                let scaled = Measurement::new(value.get() * scale as f64)?;
                Ok(scaled.get() > threshold as f64)
            }
        }
    }
}

impl From<u64> for Quantity {
    fn from(value: u64) -> Self {
        Self::Count(value)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::TokenKind;

    #[rstest]
    #[case::zero(0.0)]
    #[case::fraction(1.5)]
    #[case::largest(f64::MAX)]
    fn admits_nonnegative_finite_measurements(#[case] value: f64) {
        assert_eq!(Measurement::new(value).unwrap().get(), value);
    }

    #[rstest]
    #[case::negative(-1.0)]
    #[case::nan(f64::NAN)]
    #[case::infinity(f64::INFINITY)]
    #[case::negative_infinity(f64::NEG_INFINITY)]
    fn rejects_invalid_measurements(#[case] value: f64) {
        assert_eq!(
            Measurement::new(value),
            Err(PriceEvaluationError::InvalidQuantity)
        );
    }

    #[rstest]
    fn normalizes_milliseconds_to_the_same_duration() {
        let seconds = Quantity::measured(
            Measurement::new(1.5).unwrap(),
            QuantityUnit::Meter,
            &Meter::WallClockSecond,
        )
        .unwrap();
        let millis = Quantity::measured(
            Measurement::new(1500.0).unwrap(),
            QuantityUnit::Milliseconds,
            &Meter::WallClockSecond,
        )
        .unwrap();
        assert_eq!(seconds, millis);
    }

    #[rstest]
    #[case::token(Meter::Input(InputUnit::Token(TokenKind::Text)), QuantityUnit::Meter)]
    #[case::image(Meter::Input(InputUnit::Image), QuantityUnit::Meter)]
    #[case::character_ms(Meter::Input(InputUnit::Character), QuantityUnit::Milliseconds)]
    fn rejects_units_that_do_not_measure_the_meter(
        #[case] meter: Meter,
        #[case] unit: QuantityUnit,
    ) {
        assert_eq!(
            Quantity::measured(Measurement::new(1.5).unwrap(), unit, &meter),
            Err(PriceEvaluationError::InvalidQuantityUnit),
        );
    }

    #[rstest]
    #[case::past_float_precision(9_007_199_254_740_993, 9_007_199_254_740_992)]
    #[case::max_count(u64::MAX, u64::MAX - 1)]
    fn counts_keep_exact_thresholds(#[case] count: u64, #[case] threshold: u64) {
        assert!(Quantity::Count(count).exceeds(threshold));
        assert!(!Quantity::Count(threshold).exceeds(threshold));
        assert_eq!(
            Quantity::Count(count).exceeds_scaled(threshold, 1),
            Ok(true)
        );
    }

    #[rstest]
    fn scaled_measurement_overflow_is_rejected() {
        assert_eq!(
            Quantity::Measured(Measurement::new(f64::MAX).unwrap()).exceeds_scaled(128_000, 4),
            Err(PriceEvaluationError::InvalidQuantity),
        );
    }
}
