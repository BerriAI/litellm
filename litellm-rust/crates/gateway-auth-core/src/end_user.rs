use litellm_config::{GeneralSettings, LitellmRole};
use serde_json::Value;

use crate::value::trim;

const STANDARD_HEADERS: [&str; 2] = ["x-litellm-customer-id", "x-litellm-end-user-id"];

/// A request value that may name the end user.
#[derive(Clone, Debug, PartialEq)]
pub enum IdValue {
    Text(String),
    /// A number, already rendered as text by the host.
    Number(String),
    /// Absent, null, a boolean, or any other type.
    Unusable,
}

/// The body fields end-user resolution reads: `user`, `litellm_metadata.user`, `metadata.user_id`
/// and `safety_identifier`. The host reads a metadata field that arrives as a JSON string.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub user: IdValue,
    pub litellm_metadata_user: IdValue,
    pub metadata_user_id: IdValue,
    pub safety_identifier: IdValue,
}

#[derive(Clone, Debug, PartialEq)]
enum ConfiguredHeader {
    Customer(Box<[String]>),
    Named(String),
    Unset,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EndUserSettings {
    header: ConfiguredHeader,
    reject_structured_ids: bool,
}

impl EndUserSettings {
    /// `validate_end_user_id_in_db` is the `litellm` module setting, which also rejects text ids
    /// that decode to a JSON object or array.
    pub fn new(general: &GeneralSettings, validate_end_user_id_in_db: bool) -> Self {
        let customer: Box<[String]> = general
            .user_header_mappings
            .iter()
            .filter(|mapping| {
                mapping.litellm_user_role == LitellmRole::Customer
                    && !mapping.header_name.is_empty()
            })
            .map(|mapping| mapping.header_name.to_lowercase())
            .collect();
        let header = match general.user_header_name.as_deref() {
            _ if !customer.is_empty() => ConfiguredHeader::Customer(customer),
            Some(name) if !trim(name).is_empty() => ConfiguredHeader::Named(name.to_lowercase()),
            _ => ConfiguredHeader::Unset,
        };
        Self {
            header,
            reject_structured_ids: validate_end_user_id_in_db,
        }
    }

    fn coerce(&self, value: &IdValue) -> Option<String> {
        match value {
            IdValue::Number(text) => Some(text.clone()),
            IdValue::Text(text) => {
                let text = trim(text);
                let rejected =
                    text.is_empty() || (self.reject_structured_ids && decodes_to_container(text));
                (!rejected).then(|| text.to_owned())
            }
            IdValue::Unusable => None,
        }
    }
}

/// The end-user id a request names: the standard headers, then the configured header, then the
/// body's `user`, `litellm_metadata.user`, `metadata.user_id` and `safety_identifier`, taking the
/// first usable one. `headers` keeps every item in request order, repeats included.
pub fn resolve(
    headers: &[(String, IdValue)],
    body: &Body,
    settings: &EndUserSettings,
) -> Option<String> {
    let first_named = |name: &str| {
        headers
            .iter()
            .filter(|(header, _)| header.to_lowercase() == name)
            .find_map(|(_, value)| settings.coerce(value))
    };
    STANDARD_HEADERS
        .iter()
        .find_map(|name| first_named(name))
        .or_else(|| match &settings.header {
            ConfiguredHeader::Customer(names) => names.iter().find_map(|name| {
                headers
                    .iter()
                    .rfind(|(header, _)| header.to_lowercase() == *name)
                    .and_then(|(_, value)| settings.coerce(value))
            }),
            ConfiguredHeader::Named(name) => first_named(name),
            ConfiguredHeader::Unset => None,
        })
        .or_else(|| settings.coerce(&body.user))
        .or_else(|| settings.coerce(&body.litellm_metadata_user))
        .or_else(|| settings.coerce(&body.metadata_user_id))
        .or_else(|| settings.coerce(&body.safety_identifier))
}

fn decodes_to_container(text: &str) -> bool {
    text.starts_with(['{', '['])
        && matches!(
            serde_json::from_str(text),
            Ok(Value::Object(_) | Value::Array(_))
        )
}

#[cfg(test)]
mod tests {
    use litellm_config::{GeneralSettings, LitellmRole, UserHeaderMapping};
    use rstest::{fixture, rstest};

    use super::{Body, EndUserSettings, IdValue, resolve};

    fn text(value: &str) -> IdValue {
        IdValue::Text(value.to_owned())
    }

    fn headers(items: &[(&str, IdValue)]) -> Vec<(String, IdValue)> {
        items
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect()
    }

    fn mapping(header_name: &str, litellm_user_role: LitellmRole) -> UserHeaderMapping {
        UserHeaderMapping {
            header_name: header_name.to_owned(),
            litellm_user_role,
        }
    }

    fn settings(
        user_header_name: Option<&str>,
        mappings: Vec<UserHeaderMapping>,
        validate: bool,
    ) -> EndUserSettings {
        let general = GeneralSettings {
            user_header_name: user_header_name.map(str::to_owned),
            user_header_mappings: mappings.into(),
            ..GeneralSettings::default()
        };
        EndUserSettings::new(&general, validate)
    }

    #[fixture]
    fn empty_body() -> Body {
        Body {
            user: IdValue::Unusable,
            litellm_metadata_user: IdValue::Unusable,
            metadata_user_id: IdValue::Unusable,
            safety_identifier: IdValue::Unusable,
        }
    }

    // Expected values are what `get_end_user_id_from_request_body` in
    // litellm/proxy/auth/auth_utils.py returned before it moved here.
    #[rstest]
    #[case::standard_customer_header(&[("X-LiteLLM-Customer-Id", text("cust"))], Some("cust"))]
    #[case::standard_end_user_header(&[("x-litellm-end-user-id", text(" eu "))], Some("eu"))]
    #[case::customer_header_beats_end_user_header(
        &[("x-litellm-end-user-id", text("eu")), ("x-litellm-customer-id", text("cust"))],
        Some("cust")
    )]
    #[case::blank_standard_header_falls_through(
        &[("x-litellm-customer-id", text("  ")), ("x-litellm-end-user-id", text("eu"))],
        Some("eu")
    )]
    #[case::repeated_standard_header_takes_first_usable(
        &[("x-litellm-customer-id", text("")), ("x-litellm-customer-id", text("second"))],
        Some("second")
    )]
    #[case::numeric_header(&[("x-litellm-customer-id", IdValue::Number("42".to_owned()))], Some("42"))]
    #[case::unusable_header(&[("x-litellm-customer-id", IdValue::Unusable)], None)]
    #[case::unrelated_header(&[("x-other", text("x"))], None)]
    fn standard_headers(
        #[case] items: &[(&str, IdValue)],
        #[case] expected: Option<&str>,
        empty_body: Body,
    ) {
        let settings = settings(None, vec![], false);
        assert_eq!(
            resolve(&headers(items), &empty_body, &settings).as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::customer_mapping(
        vec![mapping("X-OpenWebUI-User-Email", LitellmRole::Customer)],
        None,
        &[("x-openwebui-user-email", text("a@b.c"))],
        Some("a@b.c")
    )]
    #[case::internal_user_mapping_ignored(
        vec![mapping("x-user", LitellmRole::InternalUser)],
        None,
        &[("x-user", text("u"))],
        None
    )]
    #[case::mapped_headers_in_mapping_order(
        vec![mapping("x-a", LitellmRole::Customer), mapping("x-b", LitellmRole::Customer)],
        None,
        &[("x-b", text("b")), ("x-a", text("a"))],
        Some("a")
    )]
    #[case::mapped_header_repeat_takes_last(
        vec![mapping("x-a", LitellmRole::Customer)],
        None,
        &[("x-a", text("first")), ("X-A", text(" "))],
        None
    )]
    #[case::mapping_wins_over_header_name(
        vec![mapping("x-a", LitellmRole::Customer)],
        Some("x-b"),
        &[("x-b", text("b"))],
        None
    )]
    #[case::header_name_when_no_customer_mapping(
        vec![mapping("x-a", LitellmRole::InternalUser)],
        Some("X-B"),
        &[("x-b", text("b"))],
        Some("b")
    )]
    #[case::empty_mapped_header_name_skipped(
        vec![mapping("", LitellmRole::Customer)],
        Some("x-b"),
        &[("x-b", text("b"))],
        Some("b")
    )]
    #[case::blank_header_name_ignored(vec![], Some("  "), &[("  ", text("b"))], None)]
    #[case::header_name_not_stripped(vec![], Some(" x-b"), &[("x-b", text("b"))], None)]
    #[case::header_name_repeat_takes_first_usable(
        vec![],
        Some("x-b"),
        &[("x-b", text("")), ("x-b", text("second"))],
        Some("second")
    )]
    fn configured_headers(
        #[case] mappings: Vec<UserHeaderMapping>,
        #[case] header_name: Option<&str>,
        #[case] items: &[(&str, IdValue)],
        #[case] expected: Option<&str>,
        empty_body: Body,
    ) {
        let settings = settings(header_name, mappings, false);
        assert_eq!(
            resolve(&headers(items), &empty_body, &settings).as_deref(),
            expected
        );
    }

    #[rstest]
    fn standard_header_beats_configured_header_and_body() {
        let settings = settings(Some("x-b"), vec![], false);
        let body = Body {
            user: text("body"),
            ..empty_body()
        };
        let items = headers(&[
            ("x-b", text("configured")),
            ("x-litellm-customer-id", text("standard")),
        ]);
        assert_eq!(
            resolve(&items, &body, &settings).as_deref(),
            Some("standard")
        );
    }

    #[rstest]
    fn configured_header_beats_body() {
        let settings = settings(Some("x-b"), vec![], false);
        let body = Body {
            user: text("body"),
            ..empty_body()
        };
        assert_eq!(
            resolve(&headers(&[("x-b", text("h"))]), &body, &settings).as_deref(),
            Some("h")
        );
    }

    #[rstest]
    #[case::user(Body { user: text(" alice "), ..empty_body() }, Some("alice"))]
    #[case::user_int(Body { user: IdValue::Number("12345".to_owned()), ..empty_body() }, Some("12345"))]
    #[case::user_beats_metadata(
        Body { user: text("u"), litellm_metadata_user: text("lm"), ..empty_body() },
        Some("u")
    )]
    #[case::blank_user_falls_through(
        Body { user: text(""), litellm_metadata_user: text("lm"), ..empty_body() },
        Some("lm")
    )]
    #[case::litellm_metadata_beats_metadata(
        Body { litellm_metadata_user: text("lm"), metadata_user_id: text("m"), ..empty_body() },
        Some("lm")
    )]
    #[case::metadata_beats_safety_identifier(
        Body { metadata_user_id: text("m"), safety_identifier: text("s"), ..empty_body() },
        Some("m")
    )]
    #[case::safety_identifier(Body { safety_identifier: text("s"), ..empty_body() }, Some("s"))]
    fn body_fields(#[case] body: Body, #[case] expected: Option<&str>) {
        let settings = settings(None, vec![], false);
        assert_eq!(resolve(&[], &body, &settings).as_deref(), expected);
    }

    #[rstest]
    #[case::json_object_kept_without_validation(
        r#"{"device_id": "d"}"#,
        false,
        Some(r#"{"device_id": "d"}"#)
    )]
    #[case::json_object_rejected_with_validation(r#"{"device_id": "d"}"#, true, None)]
    #[case::json_array_rejected_with_validation(r#" ["a","b"] "#, true, None)]
    #[case::json_scalar_kept_with_validation("[", true, Some("["))]
    #[case::invalid_json_kept_with_validation("{not json", true, Some("{not json"))]
    #[case::non_json_nan_kept_with_validation("[NaN]", true, Some("[NaN]"))]
    #[case::plain_id_kept_with_validation("alice", true, Some("alice"))]
    #[case::separators_stripped("\u{1c}\u{a0}alice\u{1f}\n", false, Some("alice"))]
    #[case::zero_width_space_kept("\u{200b}", false, Some("\u{200b}"))]
    fn text_ids(#[case] value: &str, #[case] validate: bool, #[case] expected: Option<&str>) {
        let settings = settings(None, vec![], validate);
        let body = Body {
            user: text(value),
            ..empty_body()
        };
        assert_eq!(resolve(&[], &body, &settings).as_deref(), expected);
    }
}
