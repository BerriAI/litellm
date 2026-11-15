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
    KeyIdentity = 1, SmallRecord, crate::record::KeyIdentity;
    Deployment = 2, SmallRecord, crate::record::Deployment;
    CandidateDeployments = 3, SmallRecord, serde_json::Value;
    RequestTags = 4, SmallRecord, std::vec::Vec<std::option::Option<crate::scalar::Scalar>>;
    DynamicParams = 5, SmallRecord, serde_json::Value;
    GuardrailInfo = 6, SmallRecord, std::vec::Vec<crate::record::GuardrailEntry>;
    Outcome = 7, SmallRecord, crate::record::Outcome;
    Error = 8, SmallRecord, serde_json::Value;
    Usage = 9, SmallRecord, crate::record::Usage;
    Cost = 10, SmallRecord, crate::record::Cost;
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
    KeyLimits = 30, SmallRecord, crate::record::KeyLimits;
    BudgetPlan = 31, SmallRecord, crate::record::BudgetPlan;
    BudgetReservation = 32, SmallRecord, crate::record::BudgetReservation;
    RateLimitStatuses = 33, SmallRecord, crate::record::RateLimitStatuses;
    OutputCap = 34, SmallRecord, crate::record::OutputCap;
    AttemptTiming = 35, SmallRecord, crate::record::AttemptTiming;
    RateLimitState = 36, SmallRecord, crate::record::RateLimitState;
    LoggedRequest = 37, SmallRecord, crate::record::LoggedRequest;
    McpToolCall = 38, SmallRecord, crate::record::McpToolCall;
    LogId = 39, SmallRecord, std::string::String;
    ColdStorageObjectKey = 40, SmallRecord, std::string::String;
    StandardLoggingPayloadRef = 41, Compatibility, u64;
    LoggedKeyIdentity = 42, SmallRecord, crate::record::LoggedKeyIdentity;
    LoggedModelGroup = 43, SmallRecord, crate::record::LoggedModelGroup;
    MetadataLabels = 44, SmallRecord, std::vec::Vec<crate::record::MetadataLabel>;
    InputSequenceLength = 45, SmallRecord, crate::record::InputSequenceLength;
    LimiterRequest = 46, SmallRecord, crate::record::LimiterRequest;
}

impl Field {
    pub const fn bit(self) -> u8 {
        self as u8
    }

    pub const fn derived_only(self) -> bool {
        DERIVED_ONLY.contains(self)
    }

    pub const fn by_reference(self) -> bool {
        BY_REFERENCE.contains(self)
    }
}

pub const DERIVED_ONLY: FieldSet = FieldSet::of(&[
    Field::KeyIdentity,
    Field::Deployment,
    Field::RequestTags,
    Field::Outcome,
    Field::Usage,
    Field::Cost,
    Field::KeyLimits,
    Field::BudgetPlan,
    Field::OutputCap,
    Field::AttemptTiming,
    Field::RateLimitState,
    Field::LoggedRequest,
    Field::McpToolCall,
    Field::LogId,
    Field::ColdStorageObjectKey,
    Field::StandardLoggingPayloadRef,
    Field::LoggedKeyIdentity,
    Field::LoggedModelGroup,
    Field::MetadataLabels,
    Field::InputSequenceLength,
    Field::LimiterRequest,
]);

pub const BY_REFERENCE: FieldSet = FieldSet::of(&[Field::StandardLoggingPayloadRef]);

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

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
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
    #[case::compatibility(Field::DeprecatedResponseObj, 29, "deprecated_response_obj")]
    #[case::typed(Field::KeyLimits, 30, "key_limits")]
    #[case::last(Field::LimiterRequest, 46, "limiter_request")]
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
    fn no_hook_writes_a_derived_only_field() {
        for hook in <crate::Hook as strum::VariantArray>::VARIANTS {
            assert!(
                hook.writable().intersection(DERIVED_ONLY).is_empty(),
                "{hook:?}"
            );
        }
    }

    #[rstest]
    fn a_by_reference_field_is_derived_only() {
        assert!(BY_REFERENCE.is_subset(DERIVED_ONLY));
        assert!(Field::StandardLoggingPayloadRef.by_reference());
        assert!(!Field::StandardLoggingPayload.by_reference());
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
