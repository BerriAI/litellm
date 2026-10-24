use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use litellm_cost::money::UnitPrice;
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;
use crate::vocabulary::{
    AudioFormat, EffortLevel, Endpoint, InputModality, Mode, OutputModality, VertexAiAudioApi,
};
use crate::wire::lax::Count;

macro_rules! attributes {
    ($($(#[$doc:meta])* $name:ident: $ty:ty $(as $via:ty)?),* $(,)?) => {
        /// The named, non-price fields of a catalog entry.
        #[derive(Clone, Debug, Default, PartialEq)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
        pub struct Attributes {
            $($(#[$doc])* pub $name: Option<$ty>,)*
        }

        impl Attributes {
            pub(crate) fn read(
                &mut self,
                key: &str,
                value: &Value,
            ) -> Option<Result<(), serde_json::Error>> {
                match key {
                    $(stringify!($name) => Some(
                        attributes!(@read value, $ty $(, $via)?).map(|read| self.$name = Some(read)),
                    ),)*
                    _ => None,
                }
            }
        }
    };
    (@read $value:ident, $ty:ty) => {
        <$ty as Deserialize>::deserialize($value)
    };
    (@read $value:ident, $ty:ty, $via:ty) => {
        serde_with::de::DeserializeAsWrap::<$ty, $via>::deserialize($value)
            .map(serde_with::de::DeserializeAsWrap::into_inner)
    };
}

attributes! {
    /// Other names this entry is served under.
    aliases: Vec<String>,
    audio_transcription_config: String,
    bedrock_converse_supports_strict_tools: bool,
    /// Highest reasoning effort the Bedrock output_config accepts for this model.
    bedrock_output_config_effort_ceiling: EffortLevel,
    comment: String,
    /// Reasoning effort the provider applies when the request omits reasoning_effort. Gates whether a non-default temperature or the top_p/logprobs sampling params are accepted, which hold only when the effort resolves to 'none'.
    default_reasoning_effort: EffortLevel,
    /// Date the provider deprecates the model, YYYY-MM-DD.
    deprecation_date: IsoDate,
    gemini_audio_only_live: bool,
    gemini_native_audio: bool,
    input_cost_per_token_cache_hit: UnitPrice,
    /// Databricks units, not USD.
    input_dbu_cost_per_token: Dbu,
    /// LiteLLM provider slug; one of https://docs.litellm.ai/docs/providers.
    litellm_provider: String,
    /// Maximum prompt/context tokens the model accepts.
    max_input_tokens: u64 as Count,
    /// Maximum tokens the model can generate in one response.
    max_output_tokens: u64 as Count,
    /// Legacy field: max output tokens if the provider specifies it, else max input tokens.
    max_tokens: u64 as Count,
    /// Free-form notes about the entry (e.g. pricing derivation).
    metadata: BTreeMap<String, Value>,
    /// Primary API surface / task type of the model.
    mode: Mode,
    /// Databricks units, not USD.
    output_dbu_cost_per_token: Dbu,
    /// Embedding dimension for embedding models.
    output_vector_size: u64 as Count,
    /// Smallest prefix the provider will actually cache; absent means the provider default applies.
    prompt_cache_min_tokens: u64 as Count,
    /// Provider-internal routing hints (e.g. bedrock_invocation_schema).
    provider_specific_entry: BTreeMap<String, Value>,
    /// Exact reasoning_effort levels this deployment accepts; wins over supports_* flags.
    reasoning_effort_levels: Vec<EffortLevel>,
    /// Provider default requests-per-minute limit.
    rpm: u64 as Count,
    /// Grounded search requests the provider includes for free before search is billed.
    search_free_requests: SearchFreeRequests,
    /// URL of the provider pricing/model page this entry was taken from.
    source: String,
    /// Audio container formats the model can return.
    supported_audio_formats: Vec<AudioFormat>,
    /// OpenAI-style API routes this model can be called through, e.g. /v1/chat/completions.
    supported_endpoints: Vec<Endpoint>,
    /// Input modalities the model accepts.
    supported_modalities: Vec<InputModality>,
    /// Output modalities the model can produce.
    supported_output_modalities: Vec<OutputModality>,
    /// Cloud regions the model is available in ('global' or region ids).
    supported_regions: Vec<String>,
    thinking_always_on: bool,
    token_count_framing: u64 as Count,
    token_count_model: TokenCountModel,
    token_count_tokenizer: TokenCountTokenizer,
    /// Provider default tokens-per-minute limit.
    tpm: u64 as Count,
    use_openai_responses_path: bool,
    uses_embed_content: bool,
    vertex_ai_audio_api: VertexAiAudioApi,
    /// Whether web search is billed per query or per prompt.
    web_search_billing_unit: WebSearchBillingUnit,
}

/// The model whose provider token counter counts tokens for this entry.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct TokenCountModel {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub litellm_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
}

/// Free grounded search requests per period, shared across the models of one group.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct SearchFreeRequests {
    pub count: u64,
    pub period: FreeRequestPeriod,
    pub shared_group: String,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FreeRequestPeriod {
    Day,
    Month,
}

/// Local tokenizer that counts tokens for this entry.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountTokenizer {
    KimiK3,
}

/// Whether web search is billed per query or per prompt.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchBillingUnit {
    PerQuery,
    PerPrompt,
}

/// A non-negative amount of Databricks units.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Dbu(f64);

impl Dbu {
    pub fn new(units: f64) -> Option<Self> {
        (units.is_finite() && units >= 0.0).then_some(Self(units))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Dbu {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let units = f64::deserialize(deserializer)?;
        Self::new(units).ok_or_else(|| {
            de::Error::invalid_value(
                de::Unexpected::Float(units),
                &"a non-negative number of Databricks units",
            )
        })
    }
}

/// A calendar date written YYYY-MM-DD.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
pub struct IsoDate(pub time::Date);

impl FromStr for IsoDate {
    type Err = Error;

    fn from_str(date: &str) -> Result<Self, Self::Err> {
        let mut parts = date.split('-');
        let (Some(year), Some(month), Some(day), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(Error::InvalidDate);
        };
        if year.len() != 4 || month.len() != 2 || day.len() != 2 {
            return Err(Error::InvalidDate);
        }
        let number = |part: &str| part.parse::<u8>().map_err(|_| Error::InvalidDate);
        let month = time::Month::try_from(number(month)?).map_err(|_| Error::InvalidDate)?;
        time::Date::from_calendar_date(
            year.parse().map_err(|_| Error::InvalidDate)?,
            month,
            number(day)?,
        )
        .map(Self)
        .map_err(|_| Error::InvalidDate)
    }
}

impl fmt::Display for IsoDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02}",
            self.0.year(),
            u8::from(self.0.month()),
            self.0.day()
        )
    }
}

#[cfg(feature = "schema")]
mod schema {
    use std::borrow::Cow;

    use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};

    use super::{Dbu, IsoDate};

    impl JsonSchema for IsoDate {
        fn inline_schema() -> bool {
            true
        }
        fn schema_name() -> Cow<'static, str> {
            "IsoDate".into()
        }
        fn json_schema(_: &mut SchemaGenerator) -> Schema {
            json_schema!({
                "type": "string",
                "format": "date",
                "pattern": r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
            })
        }
    }

    impl JsonSchema for Dbu {
        fn inline_schema() -> bool {
            true
        }
        fn schema_name() -> Cow<'static, str> {
            "Dbu".into()
        }
        fn json_schema(_: &mut SchemaGenerator) -> Schema {
            json_schema!({"type": "number", "minimum": 0})
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::leap_day("2028-02-29", true)]
    #[case::not_a_leap_year("2026-02-29", false)]
    #[case::impossible_day("2026-02-31", false)]
    #[case::one_digit_month("2026-2-01", false)]
    #[case::extra_part("2026-02-01-01", false)]
    fn iso_dates_are_real_calendar_dates(#[case] text: &str, #[case] valid: bool) {
        assert_eq!(text.parse::<IsoDate>().is_ok(), valid);
    }

    #[rstest]
    #[case::count_written_as_text("max_tokens", json!("8192"), true)]
    #[case::negative_count("max_tokens", json!(-1), false)]
    #[case::known_mode("mode", json!("chat"), true)]
    #[case::unknown_mode("mode", json!("chatt"), false)]
    #[case::negative_dbu("input_dbu_cost_per_token", json!(-1), false)]
    fn each_attribute_reads_its_own_type(
        #[case] key: &str,
        #[case] value: Value,
        #[case] valid: bool,
    ) {
        assert_eq!(
            Attributes::default()
                .read(key, &value)
                .map(|read| read.is_ok()),
            Some(valid)
        );
    }

    #[rstest]
    fn keys_that_are_not_attributes_are_left_to_the_caller() {
        assert!(
            Attributes::default()
                .read("input_cost_per_token", &json!(1))
                .is_none()
        );
    }
}
