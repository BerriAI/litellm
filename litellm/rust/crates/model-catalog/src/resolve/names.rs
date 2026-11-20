//! The names main's `get_model_info` tries for a model and a known provider: `get_potential_model_names` and
//! `_strip_model_name` in `litellm/utils.py`

use litellm_auth_aws::is_bedrock_region;

use super::Candidates;
use super::bedrock::{
    bedrock_base_model, bedrock_model_name_from_arn, split_bedrock_region_path,
    strip_bedrock_routing_prefix,
};

const MANTLE: &str = "bedrock_mantle";

impl Candidates {
    /// Main's seven names for `model` called through `provider`, for a provider with no model cost key of its own
    pub fn potential(model: &str, provider: &str) -> Self {
        let prefix = format!("{provider}/");
        let (split_model, combined_model_name, stripped_model_name, provider_prefixed_model_name) =
            match model.strip_prefix(&prefix) {
                Some(split) => (
                    split,
                    model.to_owned(),
                    strip_model_name(split, provider),
                    format!("{provider}/{model}"),
                ),
                None => (
                    model,
                    format!("{provider}/{model}"),
                    strip_model_name(model, provider),
                    format!("{provider}/{model}"),
                ),
            };
        let split_model = match provider {
            "bedrock" | "bedrock_converse" => {
                let region_free = split_bedrock_region_path(split_model).1;
                if region_free.starts_with("arn:") {
                    bedrock_model_name_from_arn(region_free)
                } else {
                    strip_bedrock_routing_prefix(split_model)
                }
            }
            _ => split_model,
        };
        let is_mantle = provider == MANTLE;
        let region_free_split = if is_mantle {
            without_mantle_region(split_model)
        } else {
            split_model
        };
        let combined_stripped_model_name = if is_mantle {
            format!("{MANTLE}/{}", strip_model_name(region_free_split, provider))
        } else {
            format!("{provider}/{stripped_model_name}")
        };
        let region_free_combined_model_name = if is_mantle {
            format!("{MANTLE}/{region_free_split}")
        } else {
            combined_model_name.clone()
        };
        Self {
            combined_model_name: Some(combined_model_name),
            model: model.to_owned(),
            region_free_combined_model_name: Some(region_free_combined_model_name),
            split_model: Some(region_free_split.to_owned()),
            combined_stripped_model_name: Some(combined_stripped_model_name),
            stripped_model_name: Some(stripped_model_name),
            provider_prefixed_model_name: Some(provider_prefixed_model_name),
        }
    }

    /// The names in the order `get_model_info` tries them, a missing one skipped
    pub fn in_lookup_order(&self) -> impl Iterator<Item = &str> {
        [
            self.combined_model_name.as_deref(),
            Some(self.model.as_str()),
            self.region_free_combined_model_name.as_deref(),
            self.split_model.as_deref(),
            self.combined_stripped_model_name.as_deref(),
            self.stripped_model_name.as_deref(),
            self.provider_prefixed_model_name.as_deref(),
        ]
        .into_iter()
        .flatten()
    }
}

/// Port of main's `_strip_model_name`
pub(crate) fn strip_model_name(model: &str, provider: &str) -> String {
    match provider {
        "bedrock" | "bedrock_converse" => bedrock_base_model(model).to_owned(),
        "vertex_ai" | "gemini" | "databricks" => without_digit_suffix(model).to_owned(),
        _ if model.contains("ft:") => without_finetune_suffix(model).to_owned(),
        _ => without_dated_snapshot(model).to_owned(),
    }
}

fn without_mantle_region(model: &str) -> &str {
    match model.split_once('/') {
        Some((region, rest)) if is_bedrock_region(region) => rest,
        _ => model,
    }
}

/// `re.sub(r"-\d+$", "", model)`
fn without_digit_suffix(model: &str) -> &str {
    match model.rsplit_once('-') {
        Some((head, digits)) if is_digits(digits) => head,
        _ => model,
    }
}

/// `re.sub(r"(:[^:]*){3}$", "", model)`, the organization, suffix and id of an OpenAI fine-tune
fn without_finetune_suffix(model: &str) -> &str {
    model
        .rmatch_indices(':')
        .nth(2)
        .map_or(model, |(index, _)| &model[..index])
}

/// `re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)`
fn without_dated_snapshot(model: &str) -> &str {
    let mut parts = model.rsplitn(4, '-');
    let (Some(day), Some(month), Some(year), Some(head)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return model;
    };
    let dated = [(year, 4), (month, 2), (day, 2)]
        .into_iter()
        .all(|(part, width)| part.len() == width && is_digits(part));
    if dated { head } else { model }
}

fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::dated_snapshot("claude-opus-4-5-20251101", "anthropic", "claude-opus-4-5-20251101")]
    #[case::dashed_date("gpt-4o-2024-08-06", "openai", "gpt-4o")]
    #[case::short_date_kept("gpt-4o-2024-8-06", "openai", "gpt-4o-2024-8-06")]
    #[case::vertex_digit_suffix("gemini-1.5-pro-002", "vertex_ai", "gemini-1.5-pro")]
    #[case::vertex_version_kept(
        "claude-opus-4-5@20251101",
        "vertex_ai",
        "claude-opus-4-5@20251101"
    )]
    #[case::finetune(
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        "openai",
        "ft:gpt-3.5-turbo"
    )]
    #[case::finetune_empty_suffix(
        "ft:gpt-4o-2024-08-06:my-org::id",
        "openai",
        "ft:gpt-4o-2024-08-06"
    )]
    #[case::bedrock_profile("us.anthropic.claude-opus-5-5", "bedrock", "anthropic.claude-opus-5-5")]
    fn strip_model_name_matches_main(
        #[case] model: &str,
        #[case] provider: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(strip_model_name(model, provider), expected);
    }

    #[rstest]
    #[case::bare(
        "claude-opus-5-5",
        "anthropic",
        &["anthropic/claude-opus-5-5", "claude-opus-5-5", "anthropic/claude-opus-5-5", "claude-opus-5-5",
          "anthropic/claude-opus-5-5", "claude-opus-5-5", "anthropic/claude-opus-5-5"],
    )]
    #[case::provider_prefixed(
        "anthropic/claude-sonnet-4-5-20250929",
        "anthropic",
        &["anthropic/claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4-5-20250929",
          "anthropic/claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4-5-20250929",
          "claude-sonnet-4-5-20250929", "anthropic/anthropic/claude-sonnet-4-5-20250929"],
    )]
    #[case::bedrock_invoke_profile(
        "invoke/us.anthropic.claude-opus-5-5",
        "bedrock",
        &["bedrock/invoke/us.anthropic.claude-opus-5-5", "invoke/us.anthropic.claude-opus-5-5",
          "bedrock/invoke/us.anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5",
          "bedrock/anthropic.claude-opus-5-5", "anthropic.claude-opus-5-5", "bedrock/invoke/us.anthropic.claude-opus-5-5"],
    )]
    #[case::mantle_region(
        "us-gov-west-1/anthropic.claude-opus-5-5",
        "bedrock_mantle",
        &["bedrock_mantle/us-gov-west-1/anthropic.claude-opus-5-5", "us-gov-west-1/anthropic.claude-opus-5-5",
          "bedrock_mantle/anthropic.claude-opus-5-5", "anthropic.claude-opus-5-5",
          "bedrock_mantle/anthropic.claude-opus-5-5", "us-gov-west-1/anthropic.claude-opus-5-5",
          "bedrock_mantle/us-gov-west-1/anthropic.claude-opus-5-5"],
    )]
    fn potential_names_come_in_main_lookup_order(
        #[case] model: &str,
        #[case] provider: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            Candidates::potential(model, provider)
                .in_lookup_order()
                .collect::<Vec<_>>(),
            expected
        );
    }
}
