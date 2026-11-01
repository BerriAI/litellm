use std::iter::Sum;
use std::ops::{Add, Mul};

use crate::priced::Quantity;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct UnitPrice(f64);

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Usd(f64);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Multiplier(f64);

impl UnitPrice {
    pub fn new(usd_per_unit: f64) -> Option<Self> {
        (usd_per_unit.is_finite() && usd_per_unit >= 0.0).then_some(Self(usd_per_unit))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Usd {
    pub const ZERO: Self = Self(0.0);

    pub fn new(amount: f64) -> Option<Self> {
        (amount.is_finite() && amount >= 0.0).then_some(Self(amount))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Multiplier {
    pub fn new(factor: f64) -> Option<Self> {
        (factor.is_finite() && factor >= 1.0).then_some(Self(factor))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Mul<Quantity> for UnitPrice {
    type Output = Usd;

    fn mul(self, quantity: Quantity) -> Usd {
        match quantity {
            Quantity::Count(units) => Usd(self.0 * units as f64),
            Quantity::Duration(duration) => Usd(self.0 * duration.as_secs_f64()),
        }
    }
}

impl Mul<Multiplier> for UnitPrice {
    type Output = UnitPrice;

    fn mul(self, factor: Multiplier) -> UnitPrice {
        UnitPrice(self.0 * factor.0)
    }
}

impl Mul<Multiplier> for Usd {
    type Output = Usd;

    fn mul(self, factor: Multiplier) -> Usd {
        Usd(self.0 * factor.0)
    }
}

impl Add for Usd {
    type Output = Usd;

    fn add(self, other: Usd) -> Usd {
        Usd(self.0 + other.0)
    }
}

impl Sum for Usd {
    fn sum<I: Iterator<Item = Usd>>(amounts: I) -> Usd {
        amounts.fold(Usd::ZERO, Add::add)
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use std::fmt;

    use serde::de::{self, Deserialize, Deserializer, Visitor};

    use super::{Multiplier, UnitPrice};

    #[derive(Clone, Copy)]
    struct Number(&'static str);

    impl Visitor<'_> for Number {
        type Value = f64;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> Result<f64, E> {
            Ok(value as f64)
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> Result<f64, E> {
            Ok(value as f64)
        }

        fn visit_f64<E: de::Error>(self, value: f64) -> Result<f64, E> {
            Ok(value)
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<f64, E> {
            value
                .trim()
                .parse()
                .map_err(|_| E::invalid_value(de::Unexpected::Str(value), &self))
        }
    }

    fn checked<'de, D: Deserializer<'de>, T>(
        deserializer: D,
        expected: Number,
        new: fn(f64) -> Option<T>,
    ) -> Result<T, D::Error> {
        let number = deserializer.deserialize_any(expected)?;
        new(number)
            .ok_or_else(|| de::Error::invalid_value(de::Unexpected::Float(number), &expected))
    }

    impl<'de> Deserialize<'de> for UnitPrice {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            checked(deserializer, Number("a non-negative USD amount"), Self::new)
        }
    }

    impl<'de> Deserialize<'de> for Multiplier {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            checked(
                deserializer,
                Number("a multiplier of at least 1"),
                Self::new,
            )
        }
    }
}

#[cfg(feature = "schema")]
mod schema {
    use std::borrow::Cow;

    use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};

    use super::{Multiplier, UnitPrice};

    impl JsonSchema for UnitPrice {
        fn inline_schema() -> bool {
            true
        }
        fn schema_name() -> Cow<'static, str> {
            "UnitPrice".into()
        }
        fn json_schema(_: &mut SchemaGenerator) -> Schema {
            json_schema!({"type": "number", "minimum": 0})
        }
    }

    impl JsonSchema for Multiplier {
        fn inline_schema() -> bool {
            true
        }
        fn schema_name() -> Cow<'static, str> {
            "Multiplier".into()
        }
        fn json_schema(_: &mut SchemaGenerator) -> Schema {
            json_schema!({"type": "number", "minimum": 1})
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::price(0.5)]
    #[case::free(0.0)]
    fn unit_price_keeps_finite_non_negative_values(#[case] value: f64) {
        assert_eq!(UnitPrice::new(value).map(UnitPrice::get), Some(value));
    }

    #[rstest]
    #[case::negative(-1e-9)]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    fn unit_price_rejects_what_no_price_can_be(#[case] value: f64) {
        assert_eq!(UnitPrice::new(value), None);
    }

    #[rstest]
    #[case::uplift(1.1, true)]
    #[case::neutral(1.0, true)]
    #[case::discount(0.5, false)]
    #[case::nan(f64::NAN, false)]
    fn multiplier_is_never_a_discount(#[case] value: f64, #[case] accepted: bool) {
        assert_eq!(Multiplier::new(value).is_some(), accepted);
    }

    #[rstest]
    fn amounts_follow_units_times_price_then_uplift() {
        let price = UnitPrice::new(0.25).unwrap();
        let uplift = Multiplier::new(2.0).unwrap();

        let total: Usd = [
            price * Quantity::Count(4),
            (price * uplift) * Quantity::Count(2),
        ]
        .into_iter()
        .sum();

        assert_eq!(total.get(), 2.0);
        assert_eq!((total * uplift).get(), 4.0);
    }

    #[rstest]
    fn a_duration_is_billed_in_fractional_seconds() {
        let per_second = UnitPrice::new(0.01).unwrap();

        let amount = per_second * Quantity::Duration(std::time::Duration::from_millis(2_500));

        assert!((amount.get() - 0.025).abs() < 1e-12);
    }

    #[cfg(feature = "serde")]
    #[rstest]
    #[case::number(serde_json::json!(1.5e-6), Some(1.5e-6))]
    #[case::integer(serde_json::json!(2), Some(2.0))]
    #[case::numeric_string(serde_json::json!(" 0.000002 "), Some(2e-6))]
    #[case::negative(serde_json::json!(-1), None)]
    #[case::word(serde_json::json!("free"), None)]
    #[case::boolean(serde_json::json!(true), None)]
    fn unit_price_reads_numbers_and_numeric_strings(
        #[case] written: serde_json::Value,
        #[case] expected: Option<f64>,
    ) {
        use serde::Deserialize;

        assert_eq!(
            UnitPrice::deserialize(&written).ok().map(UnitPrice::get),
            expected
        );
    }
}
