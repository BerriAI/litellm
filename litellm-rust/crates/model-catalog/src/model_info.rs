use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::{DeserializeAs, serde_as};

use crate::capabilities::{
    AudioFormat, Endpoint, InputModality, Mode, OutputModality, VertexAiAudioApi,
};
use crate::error::{InvalidDate, Problem};
use crate::prices::Prices;
use crate::pricing::{PricingFields, Usd, WebSearchBillingUnit};
use crate::rate::Rate;
use litellm_llms_types::formats::chat_completions::ReasoningEffort;
use litellm_llms_types::serde_compat::LaxI64;

pub type Fields = Map<String, Value>;

/// The model whose provider token counter counts tokens for this entry.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct TokenCountModel {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
}

/// Local tokenizer that counts tokens for this entry.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountTokenizer {
    KimiK3,
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
    type Err = InvalidDate;

    fn from_str(date: &str) -> Result<Self, Self::Err> {
        let mut parts = date.split('-');
        let (Some(year), Some(month), Some(day), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(InvalidDate);
        };
        if year.len() != 4 || month.len() != 2 || day.len() != 2 {
            return Err(InvalidDate);
        }
        let number = |part: &str| part.parse::<u8>().map_err(|_| InvalidDate);
        let month = time::Month::try_from(number(month)?).map_err(|_| InvalidDate)?;
        time::Date::from_calendar_date(year.parse().map_err(|_| InvalidDate)?, month, number(day)?)
            .map(Self)
            .map_err(|_| InvalidDate)
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
impl schemars::JsonSchema for IsoDate {
    fn inline_schema() -> bool {
        true
    }
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "IsoDate".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "format": "date",
            "pattern": r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
        })
    }
}

struct Count;

impl<'de> DeserializeAs<'de, u64> for Count {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let count = LaxI64::deserialize_as(deserializer)?;
        u64::try_from(count).map_err(|_| {
            de::Error::invalid_value(de::Unexpected::Signed(count), &"a count of zero or more")
        })
    }
}

/// The named, non-price fields of a catalog entry.
#[serde_as]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct Attributes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_transcription_config: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock_converse_supports_strict_tools: Option<bool>,
    /// Highest reasoning effort the Bedrock output_config accepts for this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock_output_config_effort_ceiling: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Reasoning effort the provider applies when the request omits reasoning_effort. Gates whether a non-default temperature or the top_p/logprobs sampling params are accepted, which hold only when the effort resolves to 'none'.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_reasoning_effort: Option<ReasoningEffort>,
    /// Date the provider deprecates the model, YYYY-MM-DD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecation_date: Option<IsoDate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gemini_audio_only_live: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gemini_native_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_cache_hit: Option<Usd>,
    /// Databricks units, not USD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_dbu_cost_per_token: Option<f64>,
    /// LiteLLM provider slug; one of https://docs.litellm.ai/docs/providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub litellm_provider: Option<String>,
    /// Maximum prompt/context tokens the model accepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub max_input_tokens: Option<u64>,
    /// Maximum tokens the model can generate in one response.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub max_output_tokens: Option<u64>,
    /// Legacy field: max output tokens if the provider specifies it, else max input tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub max_tokens: Option<u64>,
    /// Free-form notes about the entry (e.g. pricing derivation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
    /// Primary API surface / task type of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<Mode>,
    /// Databricks units, not USD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dbu_cost_per_token: Option<f64>,
    /// Embedding dimension for embedding models.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub output_vector_size: Option<u64>,
    /// Smallest prefix the provider will actually cache; absent means the provider default applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub prompt_cache_min_tokens: Option<u64>,
    /// Provider-internal routing hints (e.g. bedrock_invocation_schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_specific_entry: Option<BTreeMap<String, Value>>,
    /// Exact reasoning_effort levels this deployment accepts; wins over supports_* flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort_levels: Option<Vec<ReasoningEffort>>,
    /// Provider default requests-per-minute limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub rpm: Option<u64>,
    /// URL of the provider pricing/model page this entry was taken from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Audio container formats the model can return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_audio_formats: Option<Vec<AudioFormat>>,
    /// OpenAI-style API routes this model can be called through, e.g. /v1/chat/completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_endpoints: Option<Vec<Endpoint>>,
    /// Input modalities the model accepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_modalities: Option<Vec<InputModality>>,
    /// Output modalities the model can produce.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_output_modalities: Option<Vec<OutputModality>>,
    /// Cloud regions the model is available in ('global' or region ids).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_regions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_always_on: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub token_count_framing: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count_model: Option<TokenCountModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count_tokenizer: Option<TokenCountTokenizer>,
    /// Provider default tokens-per-minute limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde_as(deserialize_as = "Option<Count>")]
    pub tpm: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_openai_responses_path: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uses_embed_content: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_ai_audio_api: Option<VertexAiAudioApi>,
    /// Whether web search is billed per query or per prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_billing_unit: Option<WebSearchBillingUnit>,
}

/// One catalog entry read into types. Every non-null key lands in exactly one of `attributes`,
/// `prices`, `capabilities` or `unrecognized`.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub attributes: Attributes,
    pub prices: Prices,
    pub capabilities: BTreeMap<String, bool>,
    pub unrecognized: Vec<String>,
}

impl Model {
    /// Reads `fields` strictly: a known key with a bad value is a problem, an unknown key is kept
    /// in `unrecognized`, and a null is treated as absent.
    pub fn from_fields(model: &str, fields: &Fields) -> Result<Self, Vec<Problem>> {
        let attribute_keys = field_names::<Attributes>();
        let pricing_keys = field_names::<PricingFields>();
        let invalid = |key: &str, error: serde_json::Error| Problem::InvalidField {
            model: model.to_owned(),
            key: key.to_owned(),
            error,
        };

        let mut problems = Vec::new();
        let mut named = Fields::new();
        let mut rates = Vec::new();
        let mut capabilities = BTreeMap::new();
        let mut unrecognized = Vec::new();
        for (key, value) in fields.iter().filter(|(_, value)| !value.is_null()) {
            if attribute_keys.contains(&key.as_str()) || pricing_keys.contains(&key.as_str()) {
                named.insert(key.clone(), value.clone());
            } else if let Ok(rate) = key.parse::<Rate>() {
                match Usd::deserialize(value) {
                    Ok(usd) => rates.push((rate, usd)),
                    Err(error) => problems.push(invalid(key, error)),
                }
            } else if key.starts_with("supports_") {
                match bool::deserialize(value) {
                    Ok(flag) => {
                        capabilities.insert(key.clone(), flag);
                    }
                    Err(error) => problems.push(invalid(key, error)),
                }
            } else {
                unrecognized.push(key.clone());
            }
        }

        let attributes = typed::<Attributes>(&named, attribute_keys, &invalid, &mut problems);
        let pricing = typed::<PricingFields>(&named, pricing_keys, &invalid, &mut problems);
        let (Some(attributes), Some(pricing)) = (attributes, pricing) else {
            return Err(problems);
        };
        match Prices::lower(model, rates, &pricing) {
            Ok(prices) if problems.is_empty() => Ok(Self {
                attributes,
                prices,
                capabilities,
                unrecognized,
            }),
            Ok(_) => Err(problems),
            Err(lowering) => {
                problems.extend(lowering);
                Err(problems)
            }
        }
    }
}

/// `T` read from the keys of `named` it declares. On failure, each of those keys is read alone
/// so every bad one is reported by name.
fn typed<T: for<'de> Deserialize<'de>>(
    named: &Fields,
    keys: &[&str],
    invalid: &impl Fn(&str, serde_json::Error) -> Problem,
    problems: &mut Vec<Problem>,
) -> Option<T> {
    let own: Fields = named
        .iter()
        .filter(|(key, _)| keys.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    match serde_json::from_value(Value::Object(own.clone())) {
        Ok(value) => Some(value),
        Err(whole) => {
            let before = problems.len();
            problems.extend(own.into_iter().filter_map(|(key, value)| {
                serde_json::from_value::<T>(Value::Object(Fields::from_iter([(
                    key.clone(),
                    value,
                )])))
                .err()
                .map(|error| invalid(&key, error))
            }));
            if problems.len() == before {
                problems.push(invalid("", whole));
            }
            None
        }
    }
}

/// The field names a derived `Deserialize` struct declares, read from what it hands the
/// deserializer.
fn field_names<T: for<'de> Deserialize<'de>>() -> &'static [&'static str] {
    struct Probe<'a>(&'a mut &'static [&'static str]);

    impl<'de> Deserializer<'de> for Probe<'_> {
        type Error = de::value::Error;

        fn deserialize_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Self::Error> {
            Err(de::Error::custom("only structs declare field names"))
        }

        fn deserialize_struct<V: Visitor<'de>>(
            self,
            _: &'static str,
            fields: &'static [&'static str],
            _: V,
        ) -> Result<V::Value, Self::Error> {
            *self.0 = fields;
            Err(de::Error::custom("field names captured"))
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
            option unit unit_struct newtype_struct seq tuple tuple_struct map enum identifier
            ignored_any
        }
    }

    let mut fields: &'static [&'static str] = &[];
    let _ = T::deserialize(Probe(&mut fields));
    fields
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::prices::{Clock, Meter, PriceKey};
    use crate::rate::{Charge, TokenKind, Unit};

    fn fields(value: Value) -> Fields {
        let Value::Object(fields) = value else {
            panic!("entry must be an object")
        };
        fields
    }

    fn problems(entry: Value) -> Vec<String> {
        Model::from_fields("m", &fields(entry))
            .unwrap_err()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    #[rstest]
    fn every_key_lands_in_exactly_one_part() {
        let model = Model::from_fields(
            "m",
            &fields(json!({
                "litellm_provider": "openai",
                "max_tokens": "8192",
                "input_cost_per_token": "0.000001",
                "tiered_pricing": [{"range": [0, 10], "output_cost_per_token": 2e-6}],
                "supports_vision": true,
                "id": "deployment-1",
                "mode": null
            })),
        )
        .unwrap();

        assert_eq!(model.attributes.litellm_provider.as_deref(), Some("openai"));
        assert_eq!(model.attributes.max_tokens, Some(8192));
        assert_eq!(model.attributes.mode, None);
        assert_eq!(
            model.prices.get(&PriceKey {
                meter: Meter::Charge(Charge::Input(Unit::Token(TokenKind::Text))),
                tier: None,
                band: None,
                clock: Clock::Peak,
            }),
            Usd::new(1e-6)
        );
        assert_eq!(model.prices.iter().count(), 2);
        assert_eq!(
            model.capabilities,
            BTreeMap::from([("supports_vision".to_owned(), true)])
        );
        assert_eq!(model.unrecognized, vec!["id".to_owned()]);
    }

    #[rstest]
    #[case::bad_mode(json!({"mode": "chatt"}), "m.mode: unknown variant `chatt`")]
    #[case::bad_price(json!({"input_cost_per_token": "abc"}), "m.input_cost_per_token: invalid value: string \"abc\", expected a non-negative USD amount")]
    #[case::negative_price(json!({"output_cost_per_token": -1}), "m.output_cost_per_token: invalid value")]
    #[case::bad_capability(json!({"supports_vision": "yes"}), "m.supports_vision: invalid type: string \"yes\", expected a boolean")]
    #[case::negative_count(json!({"max_tokens": -5}), "m.max_tokens: invalid value: integer `-5`, expected a count of zero or more")]
    #[case::bad_date(json!({"deprecation_date": "2026-02-31"}), "m.deprecation_date: expected a calendar date YYYY-MM-DD")]
    #[case::bad_endpoint(json!({"supported_endpoints": ["/v1/telepathy"]}), "m.supported_endpoints: unknown variant `/v1/telepathy`")]
    #[case::low_uplift(json!({"regional_endpoint_uplift_multiplier": 0.5}), "m.regional_endpoint_uplift_multiplier: invalid value")]
    #[case::bad_hours(json!({"off_peak_pricing": {"hours_utc": "25:00-01:00"}}), "m.off_peak_pricing")]
    #[case::bad_weekday(json!({"off_peak_pricing": {"windows": [{"hours_utc": "00:00-01:00", "weekdays": [0]}]}}), "m.off_peak_pricing")]
    #[case::negative_guardrail(json!({"guardrail_cost_per_unit": {"unit": -1}}), "m.guardrail_cost_per_unit")]
    #[case::unknown_tier_key(json!({"tiered_pricing": [{"range": [0, 1], "typo_cost": 1}]}), "m.tiered_pricing")]
    fn a_bad_known_key_names_the_model_key_and_what_was_expected(
        #[case] entry: Value,
        #[case] expected: &str,
    ) {
        let problems = problems(entry);
        assert!(
            problems.iter().any(|problem| problem.starts_with(expected)),
            "{problems:?}"
        );
    }

    #[rstest]
    fn every_bad_key_is_reported_at_once() {
        assert_eq!(
            problems(json!({"mode": "chatt", "max_tokens": -1, "input_cost_per_token": "x"})).len(),
            3
        );
    }

    #[rstest]
    fn field_names_lists_declared_fields() {
        assert!(field_names::<Attributes>().contains(&"litellm_provider"));
        assert!(field_names::<PricingFields>().contains(&"tiered_pricing"));
        assert!(!field_names::<Attributes>().contains(&"tiered_pricing"));
    }
}
