use std::ops::{BitOr, BitOrAssign};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum FieldTier {
    SmallRecord = 1,
    Content = 2,
    Wire = 3,
    Compatibility = 4,
}

pub trait TypedField {
    const FIELD: Field;
    type Value: DeserializeOwned;
}

macro_rules! fields {
    ($($variant:ident = $bit:literal, $tier:ident, $value:ty;)*) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, strum::VariantArray,
        )]
        #[serde(rename_all = "snake_case")]
        #[repr(u8)]
        pub enum Field {
            $($variant = $bit,)*
        }

        impl Field {
            pub const fn tier(self) -> FieldTier {
                match self {
                    $(Self::$variant => FieldTier::$tier,)*
                }
            }
        }

        pub mod typed {
            $(
                pub struct $variant;

                impl super::TypedField for $variant {
                    const FIELD: super::Field = super::Field::$variant;
                    type Value = $value;
                }
            )*
        }
    };
}

fields! {
    Model = 0, SmallRecord, std::string::String;
    KeyIdentity = 1, SmallRecord, serde_json::Value;
    Deployment = 2, SmallRecord, serde_json::Value;
    CandidateDeployments = 3, SmallRecord, serde_json::Value;
    RequestTags = 4, SmallRecord, serde_json::Value;
    DynamicParams = 5, SmallRecord, serde_json::Value;
    GuardrailInfo = 6, SmallRecord, serde_json::Value;
    Outcome = 7, SmallRecord, serde_json::Value;
    Error = 8, SmallRecord, serde_json::Value;
    Usage = 9, SmallRecord, serde_json::Value;
    Cost = 10, SmallRecord, serde_json::Value;
    Metadata = 11, Content, serde_json::Value;
    Messages = 12, Content, serde_json::Value;
    Input = 13, Content, serde_json::Value;
    Tools = 14, Content, serde_json::Value;
    OptionalParams = 15, Content, serde_json::Value;
    RequestHeaders = 16, Content, serde_json::Value;
    Response = 17, Content, serde_json::Value;
    ResponseHeaders = 18, Content, serde_json::Value;
    StreamChunk = 19, Content, serde_json::Value;
    AssembledStream = 20, Content, serde_json::Value;
    ToolResult = 21, Content, serde_json::Value;
    ModelList = 22, Content, serde_json::Value;
    RequestBody = 23, Wire, serde_json::Value;
    ProviderRequestBody = 24, Wire, serde_json::Value;
    ProviderResponseRaw = 25, Wire, serde_json::Value;
    ProviderResponseHeaders = 26, Wire, serde_json::Value;
    StandardLoggingPayload = 27, Compatibility, serde_json::Value;
    DeprecatedKwargs = 28, Compatibility, serde_json::Value;
    DeprecatedResponseObj = 29, Compatibility, serde_json::Value;
}

impl Field {
    pub const fn bit(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "Vec<Field>", into = "Vec<Field>")]
pub struct FieldSet(u64);

impl FieldSet {
    pub const EMPTY: Self = Self(0);

    pub const fn of(fields: &[Field]) -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < fields.len() {
            bits |= 1 << fields[index].bit();
            index += 1;
        }
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn contains(self, field: Field) -> bool {
        self.0 & (1 << field.bit()) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_subset(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub fn iter(self) -> impl Iterator<Item = Field> {
        <Field as strum::VariantArray>::VARIANTS
            .iter()
            .copied()
            .filter(move |field| self.contains(*field))
    }
}

impl BitOr for FieldSet {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOrAssign for FieldSet {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl From<Vec<Field>> for FieldSet {
    fn from(fields: Vec<Field>) -> Self {
        Self::of(&fields)
    }
}

impl From<FieldSet> for Vec<Field> {
    fn from(set: FieldSet) -> Self {
        set.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::first(Field::Model, 0, "model")]
    #[case::wire(Field::ProviderRequestBody, 24, "provider_request_body")]
    #[case::last(Field::DeprecatedResponseObj, 29, "deprecated_response_obj")]
    fn a_field_keeps_its_bit_and_its_snake_case_name(
        #[case] field: Field,
        #[case] bit: u8,
        #[case] name: &str,
    ) {
        assert_eq!(field.bit(), bit);
        assert_eq!(serde_json::to_value(field).unwrap(), json!(name));
        assert_eq!(serde_json::from_value::<Field>(json!(name)).unwrap(), field);
    }

    #[rstest]
    fn a_field_set_round_trips_as_a_list_of_names() {
        let set = FieldSet::of(&[Field::Usage, Field::Model]);
        assert_eq!(
            serde_json::to_value(set).unwrap(),
            json!(["model", "usage"])
        );
        assert_eq!(
            serde_json::from_value::<FieldSet>(json!(["usage", "model"])).unwrap(),
            set
        );
        assert_eq!(set.bits(), 0b10_0000_0001);
    }

    #[rstest]
    fn subset_and_difference_are_bit_operations() {
        let small = FieldSet::of(&[Field::Model]);
        let large = FieldSet::of(&[Field::Model, Field::Metadata]);
        assert!(small.is_subset(large));
        assert!(!large.is_subset(small));
        assert_eq!(large.difference(small), FieldSet::of(&[Field::Metadata]));
    }
}
