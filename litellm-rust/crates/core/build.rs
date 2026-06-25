use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ProviderEndpointSupportInput {
    providers: BTreeMap<String, ProviderInput>,
    #[serde(default)]
    default_creds: BTreeMap<String, ProviderDefaultCredsInput>,
}

#[derive(Debug, Deserialize)]
struct ProviderInput {
    display_name: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct ProviderDefaultCredsInput {
    default_api_base: Option<String>,
    api_key_env_var: Option<String>,
}

#[derive(Debug)]
struct ProviderMetadataInput {
    routing_name: String,
    display_name: String,
    docs_url: String,
    default_api_base: Option<String>,
    api_key_env_var: Option<String>,
}

fn rust_string(value: &str) -> String {
    format!("{value:?}")
}

fn rust_option(value: Option<&str>) -> String {
    value
        .map(|value| format!("Some({})", rust_string(value)))
        .unwrap_or_else(|| "None".to_string())
}

fn variant_name(routing_name: &str) -> String {
    routing_name
        .split(['_', '-', '.', '/'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    let mut out = String::new();
                    out.extend(first.to_uppercase());
                    out.push_str(chars.as_str());
                    out
                }
                None => String::new(),
            }
        })
        .collect()
}

fn provider_metadata_from_endpoint_support(
    registry: ProviderEndpointSupportInput,
) -> Vec<ProviderMetadataInput> {
    let ProviderEndpointSupportInput {
        providers,
        default_creds,
    } = registry;

    providers
        .into_iter()
        .map(|(routing_name, provider)| {
            let default_creds = default_creds.get(&routing_name);
            ProviderMetadataInput {
                routing_name,
                display_name: provider.display_name,
                docs_url: provider.url,
                default_api_base: default_creds.and_then(|creds| creds.default_api_base.clone()),
                api_key_env_var: default_creds.and_then(|creds| creds.api_key_env_var.clone()),
            }
        })
        .collect()
}

fn validate_providers(providers: &[ProviderMetadataInput]) {
    let mut routing_names = HashSet::new();
    let mut variants = HashSet::new();

    for provider in providers {
        if provider.routing_name.trim().is_empty() {
            panic!("provider routing_name cannot be empty");
        }
        if provider.display_name.trim().is_empty() {
            panic!(
                "provider {} has an empty display_name",
                provider.routing_name
            );
        }
        if provider.docs_url.trim().is_empty() {
            panic!("provider {} has an empty docs_url", provider.routing_name);
        }
        if !routing_names.insert(provider.routing_name.as_str()) {
            panic!("duplicate provider routing_name: {}", provider.routing_name);
        }

        let variant = variant_name(&provider.routing_name);
        if variant.is_empty() {
            panic!(
                "provider {} generated an empty Rust variant",
                provider.routing_name
            );
        }
        if variant.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
            panic!(
                "provider {} generated Rust variant {variant} starting with a digit",
                provider.routing_name
            );
        }
        if !variants.insert(variant.clone()) {
            panic!(
                "provider {} generated duplicate Rust variant {variant}",
                provider.routing_name
            );
        }
    }
}

fn generate_provider_code(providers: &[ProviderMetadataInput]) -> String {
    let variants: Vec<String> = providers
        .iter()
        .map(|provider| variant_name(&provider.routing_name))
        .collect();

    let enum_variants = variants
        .iter()
        .map(|variant| format!("    {variant},"))
        .collect::<Vec<_>>()
        .join("\n");

    let all_values = variants
        .iter()
        .map(|variant| format!("        LlmProvider::{variant},"))
        .collect::<Vec<_>>()
        .join("\n");

    let metadata_values = providers
        .iter()
        .zip(variants.iter())
        .map(|(provider, variant)| {
            format!(
                "        ProviderMetadata {{ provider: LlmProvider::{variant}, routing_name: {}, display_name: {}, docs_url: {}, default_api_base: {}, api_key_env_var: {} }},",
                rust_string(&provider.routing_name),
                rust_string(&provider.display_name),
                rust_string(&provider.docs_url),
                rust_option(provider.default_api_base.as_deref()),
                rust_option(provider.api_key_env_var.as_deref()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let metadata_match_arms = variants
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            format!("            LlmProvider::{variant} => &Self::METADATA[{index}],")
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"// @generated by crates/core/build.rs from provider_endpoints_support.json.
// Do not edit this file by hand.

use std::fmt;
use std::str::FromStr;

use crate::error::CoreError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderMetadata {{
    pub provider: LlmProvider,
    pub routing_name: &'static str,
    pub display_name: &'static str,
    pub docs_url: &'static str,
    pub default_api_base: Option<&'static str>,
    pub api_key_env_var: Option<&'static str>,
}}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmProvider {{
{enum_variants}
}}

impl LlmProvider {{
    pub const ALL: [LlmProvider; {provider_count}] = [
{all_values}
    ];

    pub const METADATA: [ProviderMetadata; {provider_count}] = [
{metadata_values}
    ];

    pub fn metadata(self) -> &'static ProviderMetadata {{
        match self {{
{metadata_match_arms}
        }}
    }}

    pub fn as_str(self) -> &'static str {{
        self.metadata().routing_name
    }}

    pub fn display_name(self) -> &'static str {{
        self.metadata().display_name
    }}

    pub fn docs_url(self) -> &'static str {{
        self.metadata().docs_url
    }}

    pub fn default_api_base(self) -> Option<&'static str> {{
        self.metadata().default_api_base
    }}

    pub fn api_key_env_var(self) -> Option<&'static str> {{
        self.metadata().api_key_env_var
    }}
}}

impl fmt::Display for LlmProvider {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        f.write_str(self.as_str())
    }}
}}

impl FromStr for LlmProvider {{
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {{
        LlmProvider::ALL
            .iter()
            .copied()
            .find(|provider| provider.as_str() == value)
            .ok_or_else(|| CoreError::InvalidProvider(value.to_string()))
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    fn endpoint_support_provider_values() -> Vec<String> {{
        let raw = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../provider_endpoints_support.json"
        ));
        let registry = serde_json::from_str::<serde_json::Value>(raw)
            .expect("provider_endpoints_support.json parses");
        let mut values = registry
            .get("providers")
            .and_then(|providers| providers.as_object())
            .expect("provider_endpoints_support.json has providers object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        values.sort();
        values
    }}

    #[test]
    fn provider_values_match_endpoint_support_registry() {{
        let registry_values = endpoint_support_provider_values();
        assert_eq!(LlmProvider::ALL.len(), registry_values.len());
        assert_eq!(
            LlmProvider::ALL
                .iter()
                .map(|provider| provider.as_str().to_string())
                .collect::<Vec<_>>(),
            registry_values
        );
    }}

    #[test]
    fn from_str_round_trips_all_providers() {{
        for provider in LlmProvider::ALL {{
            assert_eq!(LlmProvider::from_str(provider.as_str()), Ok(provider));
            assert_eq!(provider.to_string(), provider.as_str());
            assert!(!provider.docs_url().is_empty());
        }}
    }}

    #[test]
    fn from_str_rejects_unknown_provider() {{
        assert_eq!(
            LlmProvider::from_str("not-a-provider"),
            Err(CoreError::InvalidProvider("not-a-provider".to_string()))
        );
    }}

    #[test]
    fn provider_metadata_exposes_optional_defaults() {{
        assert_eq!(
            LlmProvider::Mistral.display_name(),
            "Mistral AI API (`mistral`)"
        );
        assert_eq!(
            LlmProvider::Mistral.docs_url(),
            "https://docs.litellm.ai/docs/providers/mistral"
        );
        assert_eq!(
            LlmProvider::Mistral.default_api_base(),
            Some("https://api.mistral.ai/v1")
        );
        assert_eq!(LlmProvider::Mistral.api_key_env_var(), None);
    }}
}}
"#,
        enum_variants = enum_variants,
        provider_count = providers.len(),
        all_values = all_values,
        metadata_values = metadata_values,
        metadata_match_arms = metadata_match_arms,
    )
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let providers_path = manifest_dir.join("../../../provider_endpoints_support.json");
    println!("cargo:rerun-if-changed={}", providers_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("build.rs").display()
    );

    let raw = fs::read_to_string(&providers_path).unwrap_or_else(|err| {
        panic!(
            "failed to read provider endpoint support registry {}: {err}",
            providers_path.display()
        )
    });
    let registry: ProviderEndpointSupportInput = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", providers_path.display()));
    let providers = provider_metadata_from_endpoint_support(registry);
    validate_providers(&providers);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(
        out_dir.join("provider_generated.rs"),
        generate_provider_code(&providers),
    )
    .expect("failed to write generated provider code");
}
