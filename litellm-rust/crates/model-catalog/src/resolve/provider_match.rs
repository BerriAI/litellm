/// main's `_check_provider_match`. A row with no provider matches any request, and a few requested providers reuse
/// another provider's rows
pub fn provider_matches(row_provider: Option<&str>, requested: Option<&str>) -> bool {
    let (Some(row), Some(requested)) = (
        row_provider,
        requested.filter(|requested| !requested.is_empty()),
    ) else {
        return true;
    };
    if row == requested {
        return true;
    }
    let same_family = |family: &str| requested == family && row.starts_with(family);
    same_family("vertex_ai")
        || same_family("fireworks_ai")
        || (requested.starts_with("bedrock") && row.starts_with("bedrock"))
        || matches!(requested, "litellm_proxy" | "github" | "nadir")
        || (requested == "azure_ai" && matches!(row, "azure" | "openai"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::no_row_provider(None, Some("openai"), true)]
    #[case::no_request(Some("openai"), None, true)]
    #[case::empty_request(Some("openai"), Some(""), true)]
    #[case::same(Some("openai"), Some("openai"), true)]
    #[case::other(Some("anthropic"), Some("openai"), false)]
    #[case::vertex_family(Some("vertex_ai-language-models"), Some("vertex_ai"), true)]
    #[case::vertex_family_is_one_way(Some("vertex_ai"), Some("vertex_ai-language-models"), false)]
    #[case::fireworks_family(Some("fireworks_ai-embedding-models"), Some("fireworks_ai"), true)]
    #[case::bedrock_family(Some("bedrock"), Some("bedrock_converse"), true)]
    #[case::proxy_reuses_anything(Some("anthropic"), Some("litellm_proxy"), true)]
    #[case::github_reuses_anything(Some("anthropic"), Some("github"), true)]
    #[case::nadir_reuses_anything(Some("anthropic"), Some("nadir"), true)]
    #[case::azure_ai_reuses_azure(Some("azure"), Some("azure_ai"), true)]
    #[case::azure_ai_reuses_openai(Some("openai"), Some("azure_ai"), true)]
    #[case::azure_ai_not_anthropic(Some("anthropic"), Some("azure_ai"), false)]
    fn follows_mains_table(
        #[case] row: Option<&str>,
        #[case] requested: Option<&str>,
        #[case] expected: bool,
    ) {
        assert_eq!(provider_matches(row, requested), expected);
    }
}
