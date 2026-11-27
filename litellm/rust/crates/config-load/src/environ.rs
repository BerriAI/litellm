use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_config::{Object, Value};
use litellm_router_types::holds_secret_pointer;
use litellm_secrets::{Secret, SecretResolver};
use serde_yaml_ng::Mapping;

use crate::{ConfigEnvironment, Error, environment_values};

const PREFIX: &str = "os.environ/";
/// `DEFAULT_MAX_RECURSE_DEPTH` in litellm/constants.py.
const MAX_DEPTH: usize = 100;
const ENVIRONMENT_VARIABLES: &str = "environment_variables";

/// Every `os.environ/` reference in `document`, read the way `ProxyConfig._check_for_os_environ_vars`
/// reads it. `environment_variables` is resolved first, against `secrets` alone, and then sits
/// in front of the environment `secrets` reads for the rest of the document. Also returns that
/// layered resolver, which reads a reference the way the rest of the document was read.
pub(crate) async fn resolve(
    document: Value,
    secrets: SecretResolver,
) -> Result<(Value, SecretResolver), Error> {
    let Value::Mapping(mut document) = document else {
        return Ok((document, secrets));
    };
    let key = Value::String(ENVIRONMENT_VARIABLES.into());
    let section = match document.remove(&key) {
        Some(section) => Some(resolve_value(&key, section, 0, &secrets).await?),
        None => None,
    };
    let values = match &section {
        Some(section @ Value::Mapping(_)) => {
            environment_values(&serde_yaml_ng::from_value::<Object>(section.clone())?)?
        }
        _ => Default::default(),
    };
    let environment = Arc::new(ConfigEnvironment::new(values, secrets.environment()));
    let layered = secrets.with_environment(environment);
    let mut document = resolve_mapping(document, 0, &layered).await?;
    if let Some(section) = section {
        document.insert(key, section);
    }
    Ok((Value::Mapping(document), layered))
}

fn resolve_mapping<'a>(
    mapping: Mapping,
    depth: usize,
    secrets: &'a SecretResolver,
) -> BoxFuture<'a, Result<Mapping, Error>> {
    Box::pin(async move {
        if depth > MAX_DEPTH {
            tracing::warn!(
                max_depth = MAX_DEPTH,
                "maximum recursion depth reached while processing config"
            );
            return Ok(mapping);
        }
        let mut resolved = Mapping::with_capacity(mapping.len());
        for (key, value) in mapping {
            let value = resolve_value(&key, value, depth, secrets).await?;
            resolved.insert(key, value);
        }
        Ok(resolved)
    })
}

async fn resolve_value(
    key: &Value,
    value: Value,
    depth: usize,
    secrets: &SecretResolver,
) -> Result<Value, Error> {
    match value {
        Value::Mapping(mapping) => Ok(Value::Mapping(
            resolve_mapping(mapping, depth + 1, secrets).await?,
        )),
        Value::Sequence(items) => {
            let mut resolved = Vec::with_capacity(items.len());
            for item in items {
                resolved.push(match item {
                    Value::Mapping(mapping) => {
                        Value::Mapping(resolve_mapping(mapping, depth + 1, secrets).await?)
                    }
                    item => item,
                });
            }
            Ok(Value::Sequence(resolved))
        }
        Value::String(text)
            if text.starts_with(PREFIX) && !key.as_str().is_some_and(holds_secret_pointer) =>
        {
            read(&text, secrets).await
        }
        value => Ok(value),
    }
}

/// One `os.environ/` reference, `reference` holding the prefix, read as the walk reads it.
pub(crate) async fn read(reference: &str, secrets: &SecretResolver) -> Result<Value, Error> {
    let name = &reference[PREFIX.len()..];
    let secret = secrets
        .get_secret(name, None)
        .await
        .map_err(|source| Error::Secret {
            reference: reference.to_owned(),
            source,
        })?;
    Ok(match secret {
        None => Value::Null,
        Some(Secret::String(value)) => Value::String(value.expose().to_owned()),
        Some(Secret::Bool(value)) => Value::Bool(value),
        Some(Secret::Json(value)) => serde_yaml_ng::to_value(value)?,
    })
}

#[cfg(test)]
mod tests {
    use litellm_secrets::{OidcResolver, SecretManagerState};
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn environment() -> SecretResolver {
        SecretResolver::new_python_compatible(
            Arc::new(SecretManagerState::default()),
            Arc::new(|name: &str| {
                let value = match name {
                    "PROBE_SET" => "set-value",
                    "PROBE_TRUE" => "True",
                    "PROBE_FALSE" => "false",
                    "PROBE_EMPTY" => "",
                    "PROBE_NUM" => "42",
                    "PROBE_SPACE" => " true ",
                    "PROBE_YES" => "yes",
                    _ => return None,
                };
                Some(value.to_owned())
            }),
            OidcResolver::new(litellm_http::Client::for_test()),
        )
    }

    fn yaml(text: &str) -> Value {
        serde_yaml_ng::from_str(text).unwrap()
    }

    fn nested(depth: usize) -> Value {
        (0..depth).fold(yaml("leaf: os.environ/PROBE_SET"), |inner, _| {
            Value::Mapping(Mapping::from_iter([(Value::String("k".into()), inner)]))
        })
    }

    fn leaf(document: &Value) -> &Value {
        match document.get("k") {
            Some(inner) => leaf(inner),
            None => &document["leaf"],
        }
    }

    // Expected values are what `ProxyConfig()._check_for_os_environ_vars` returns for the same
    // input and environment.
    #[rstest]
    #[case::set("a: os.environ/PROBE_SET", "a: set-value")]
    #[case::unset("a: os.environ/PROBE_UNSET", "a: null")]
    #[case::capitalized_true("a: os.environ/PROBE_TRUE", "a: true")]
    #[case::false_text("a: os.environ/PROBE_FALSE", "a: false")]
    #[case::padded_true("a: os.environ/PROBE_SPACE", "a: true")]
    #[case::empty("a: os.environ/PROBE_EMPTY", "a: ''")]
    #[case::number_stays_text("a: os.environ/PROBE_NUM", "a: '42'")]
    #[case::yes_stays_text("a: os.environ/PROBE_YES", "a: 'yes'")]
    #[case::bare_prefix("a: os.environ/", "a: null")]
    #[case::not_a_prefix(
        "{a: ' os.environ/PROBE_SET', b: OS.ENVIRON/PROBE_SET}",
        "{a: ' os.environ/PROBE_SET', b: OS.ENVIRON/PROBE_SET}"
    )]
    #[case::only_mappings_inside_lists(
        "a: [os.environ/PROBE_SET, {b: os.environ/PROBE_SET}, [os.environ/PROBE_SET]]",
        "a: [os.environ/PROBE_SET, {b: set-value}, [os.environ/PROBE_SET]]"
    )]
    #[case::nested("a: {b: {c: os.environ/PROBE_SET}}", "a: {b: {c: set-value}}")]
    #[case::secret_pointers(
        "{anthropic_issuer_signing_key_ref: os.environ/PROBE_SET, anthropic_keycloak_client_secret_ref: os.environ/PROBE_SET, x: {anthropic_issuer_signing_key_ref: os.environ/PROBE_SET}}",
        "{anthropic_issuer_signing_key_ref: os.environ/PROBE_SET, anthropic_keycloak_client_secret_ref: os.environ/PROBE_SET, x: {anthropic_issuer_signing_key_ref: os.environ/PROBE_SET}}"
    )]
    #[case::non_strings("{a: 1, b: true, c: null, d: 1.5}", "{a: 1, b: true, c: null, d: 1.5}")]
    #[case::environment_variables_read_the_process_environment(
        "environment_variables: {CFG_ONLY: from-config, CFG_REF: os.environ/PROBE_SET}",
        "environment_variables: {CFG_ONLY: from-config, CFG_REF: set-value}"
    )]
    #[tokio::test]
    async fn resolves_like_python(
        environment: SecretResolver,
        #[case] input: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            resolve(yaml(input), environment).await.unwrap().0,
            yaml(expected)
        );
    }

    #[rstest]
    #[case::at_the_limit(100, "set-value")]
    #[case::past_the_limit(101, "os.environ/PROBE_SET")]
    #[tokio::test]
    async fn stops_at_the_depth_limit(
        environment: SecretResolver,
        #[case] depth: usize,
        #[case] expected: &str,
    ) {
        let resolved = resolve(nested(depth), environment).await.unwrap().0;
        assert_eq!(leaf(&resolved), &Value::String(expected.into()));
    }

    // Python resolves the document before it exports `environment_variables`, so on a first load
    // these references read only the process environment. The gateway reads them the way a later
    // load does, once the variables are exported.
    #[rstest]
    #[case::exported_value("CFG_ONLY", "from-config")]
    #[case::exported_reference("CFG_REF", "set-value")]
    #[case::exported_bool("CFG_BOOL", "true")]
    #[case::config_wins_over_the_process("PROBE_SET", "from-config")]
    #[case::unset_reference_exports_nothing("CFG_UNSET", "~")]
    #[case::oidc_reads_the_exported_value("oidc/env/CFG_ONLY", "from-config")]
    #[case::oidc_reads_config_before_the_process("oidc/env/PROBE_SET", "from-config")]
    #[tokio::test]
    async fn environment_variables_come_before_the_process_environment(
        environment: SecretResolver,
        #[case] name: &str,
        #[case] expected: &str,
    ) {
        let document = yaml(&format!(
            "environment_variables: {{CFG_ONLY: from-config, CFG_REF: os.environ/PROBE_SET, CFG_BOOL: 'TRUE', PROBE_SET: from-config, CFG_UNSET: os.environ/PROBE_UNSET}}\nmodel_list: [{{litellm_params: {{api_key: os.environ/{name}}}}}]"
        ));
        let resolved = resolve(document, environment).await.unwrap().0;
        assert_eq!(
            resolved["model_list"][0]["litellm_params"]["api_key"],
            yaml(expected)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn rejects_a_nested_environment_variable(environment: SecretResolver) {
        let error = resolve(yaml("environment_variables: {NESTED: {a: b}}"), environment)
            .await
            .err()
            .expect("a nested variable is rejected");
        assert!(matches!(error, Error::Environment { name } if name == "NESTED"));
    }
}
