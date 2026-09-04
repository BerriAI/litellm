use serde_json::{Map, Value};

use crate::error::Error;

pub use super::aws_base::{
    aws_auth_config, aws_signature_headers, host_supplied_credentials, is_sigv4_computed_header,
    resolve_credentials, sign_bedrock_post,
};
use super::aws_base::{bedrock_model_id_and_region, resolve_bedrock_region};
use super::constants::{AWS_BEARER_TOKEN_BEDROCK, BEDROCK_SERVICE};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BedrockAuth {
    Bearer(String),
    SigV4 {
        region: String,
        service: &'static str,
    },
}

pub fn resolve_bedrock_auth(
    api_key: Option<&str>,
    model: &str,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<BedrockAuth, Error> {
    let bearer = api_key
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(AWS_BEARER_TOKEN_BEDROCK).filter(|value| !value.trim().is_empty()));
    if let Some(token) = bearer {
        return Ok(BedrockAuth::Bearer(token));
    }
    let (_, model_region) = bedrock_model_id_and_region(model);
    Ok(BedrockAuth::SigV4 {
        region: resolve_bedrock_region(model_region.as_deref(), optional_params, env_lookup),
        service: BEDROCK_SERVICE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_explicit_bearer_uses_environment_before_sigv4() {
        let env = |name: &str| {
            (name == AWS_BEARER_TOKEN_BEDROCK).then(|| "environment-token".to_string())
        };
        assert_eq!(
            resolve_bedrock_auth(Some("  "), "model", &Map::new(), &env).unwrap(),
            BedrockAuth::Bearer("environment-token".to_string())
        );
    }

    #[test]
    fn missing_bearer_selects_body_aware_sigv4() {
        assert_eq!(
            resolve_bedrock_auth(None, "us-west-2/model", &Map::new(), &|_| None).unwrap(),
            BedrockAuth::SigV4 {
                region: "us-west-2".to_string(),
                service: BEDROCK_SERVICE,
            }
        );
    }
}
