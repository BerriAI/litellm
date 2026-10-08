use litellm_core_utils::call_arguments::CallArguments;
use serde::Serialize;
use serde_json::Value;

use crate::{Error, is_owned};

pub fn compose_body<B: Serialize>(
    arguments: &CallArguments,
    body: &B,
    consumed: &[&str],
) -> Result<Value, Error> {
    let Value::Object(fields) = serde_json::to_value(body).map_err(|_| Error::Body)? else {
        return Err(Error::Body);
    };
    let overrides = match arguments.get("extra_body") {
        None | Some(Value::Null) => None,
        Some(Value::Object(fields)) => Some(fields),
        Some(_) => return Err(Error::ExtraBody),
    };
    let extensions = arguments
        .iter()
        .filter(|(name, _)| !consumed.contains(&name.as_str()));
    Ok(Value::Object(
        fields
            .into_iter()
            .chain(
                extensions
                    .chain(overrides.into_iter().flatten())
                    .filter(|(name, _)| !is_owned(name))
                    .map(|(name, value)| (name.clone(), value.clone())),
            )
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[test]
    fn composition_preserves_extensions_and_applies_shallow_explicit_overrides() {
        let original = json!({
            "known": false, "future": {"old": 1}, "null": null, "zero": 0,
            "metadata": {"host": true}, "_litellm_trace_id": "t", "timeout": 30, "api_key": "secret",
            "aws_secret_access_key": "secret", "vertex_project": "p",
            "extra_body": {
                "known": null, "future": {"new": [false, 0, null]},
                "metadata": {"provider": true}, "model": "ignored", "api_key": "ignored"
            }
        });
        let arguments = serde_json::from_value(original.clone()).unwrap();
        let body = compose_body(
            &arguments,
            &json!({"model":"resolved", "known":false}),
            &["known"],
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "model":"resolved", "known":null, "future":{"new":[false,0,null]},
                "null":null, "zero":0
            })
        );
        assert_eq!(serde_json::to_value(arguments).unwrap(), original);
    }

    #[rstest]
    #[case::boolean(json!(false))]
    #[case::number(json!(0))]
    #[case::array(json!([]))]
    #[case::string(json!(""))]
    fn invalid_extra_body_is_rejected_without_coercing_it_to_empty(#[case] value: Value) {
        let arguments = serde_json::from_value(json!({"extra_body":value})).unwrap();
        assert_eq!(
            compose_body(&arguments, &json!({}), &[]),
            Err(Error::ExtraBody)
        );
    }

    #[test]
    fn null_extra_body_is_coerced_to_empty_object() {
        let arguments = serde_json::from_value(json!({"extra_body":null})).unwrap();
        assert_eq!(
            compose_body(&arguments, &json!({}), &[]).unwrap(),
            json!({})
        );
    }

    #[test]
    fn a_non_object_body_is_rejected() {
        let arguments = serde_json::from_value(json!({})).unwrap();
        assert_eq!(compose_body(&arguments, &json!([1]), &[]), Err(Error::Body));
    }
}
