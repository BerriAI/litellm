use crate::base_llm::endpoint::ResolvedEndpoint;
use litellm_core_utils::{
    ApiUrlError,
    url_utils::{ApiUrl, Complete, EndpointTarget, Mount, QueryParameter, check_segment},
};
use litellm_llms_types::endpoint::EndpointPath;

crate::provider_endpoints! {
    pub enum AzureEndpoint {
        Ocr => POST "/providers/mistral/azure/ocr",
        CohereParse => POST "/providers/cohere/v2/parse",
        Messages => POST "/anthropic/v1/messages",
    }
}

#[derive(Clone, Debug)]
pub struct AnalyzeDocument {
    model: String,
}

impl AnalyzeDocument {
    const PREFIX: EndpointPath = EndpointPath::new("/documentintelligence/documentModels");

    pub fn new(model: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            model: check_segment(model)?.to_owned(),
        })
    }

    fn action(&self) -> String {
        format!("{}:analyze", self.model)
    }

    pub fn matches(&self, path: &str) -> Result<bool, ApiUrlError> {
        let action = encoded_action(&self.action())?;
        Ok(Self::PREFIX
            .segments()
            .chain(std::iter::once(action.as_str()))
            .eq(path.split('/').filter(|segment| !segment.is_empty())))
    }

    pub fn legacy_target(&self, value: &str) -> Result<EndpointTarget, ApiUrlError> {
        let exact = ApiUrl::<Complete>::parse_exact(value)?;
        let suffix = format!(
            "/{}/{}",
            Self::PREFIX.segments().collect::<Vec<_>>().join("/"),
            encoded_action(&self.action())?
        );
        if exact
            .as_url()
            .path()
            .trim_end_matches('/')
            .ends_with(&suffix)
        {
            return Ok(EndpointTarget::Exact(exact));
        }
        Ok(EndpointTarget::Mount(ApiUrl::<Mount>::parse_mount(value)?))
    }

    pub fn resolve_legacy(
        &self,
        value: &str,
        query: &[QueryParameter<'_>],
    ) -> Result<ResolvedEndpoint, ApiUrlError> {
        let target = self.legacy_target(value)?;
        if let EndpointTarget::Exact(_) = target {
            let completed =
                ApiUrl::<Mount>::parse_mount(value)?.resolve_segments(std::iter::empty(), query)?;
            return Ok(ResolvedEndpoint::new(reqwest::Method::POST, completed));
        }
        let parsed = ApiUrl::<Complete>::parse_exact(value)?;
        let path = parsed.as_url().path().trim_end_matches('/');
        let mount = ApiUrl::<Mount>::parse_mount(value)?;
        let suffix = if path.ends_with("/documentintelligence/documentModels") {
            vec![self.action()]
        } else if path.ends_with("/documentintelligence") {
            vec!["documentModels".into(), self.action()]
        } else {
            return self.resolve_with_query(&target, query);
        };
        Ok(ResolvedEndpoint::new(
            reqwest::Method::POST,
            mount.resolve_segments(suffix.iter().map(String::as_str), query)?,
        ))
    }

    pub fn resolve_with_query(
        &self,
        target: &EndpointTarget,
        query: &[QueryParameter<'_>],
    ) -> Result<ResolvedEndpoint, ApiUrlError> {
        let url = match target {
            EndpointTarget::Exact(url) => url.clone(),
            EndpointTarget::Mount(mount) => mount.resolve_segments(
                Self::PREFIX
                    .segments()
                    .chain(std::iter::once(self.action().as_str())),
                query,
            )?,
        };
        Ok(ResolvedEndpoint::new(reqwest::Method::POST, url))
    }
}

impl crate::base_llm::endpoint::ProviderEndpoint for AnalyzeDocument {
    fn resolve(&self, target: &EndpointTarget) -> Result<ResolvedEndpoint, ApiUrlError> {
        self.resolve_with_query(target, &[])
    }
}

fn encoded_action(action: &str) -> Result<String, ApiUrlError> {
    let mount = ApiUrl::<Mount>::parse_mount("https://encoding.invalid")?;
    Ok(mount.resolve_segments([action], &[])?.as_url().path()[1..].to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_core_utils::url_utils::QueryPolicy;
    use rstest::rstest;

    #[rstest]
    #[case::root("", "/documentintelligence/documentModels/model:analyze")]
    #[case::prefix(
        "/documentintelligence",
        "/documentintelligence/documentModels/model:analyze"
    )]
    #[case::models(
        "/documentintelligence/documentModels",
        "/documentintelligence/documentModels/model:analyze"
    )]
    #[case::complete(
        "/documentintelligence/documentModels/model:analyze",
        "/documentintelligence/documentModels/model:analyze"
    )]
    fn legacy_query_defaults_are_applied_once(#[case] suffix: &str, #[case] path: &str) {
        let operation = AnalyzeDocument::new("model").unwrap();
        let query = [QueryParameter {
            key: "api-version",
            value: "configured",
            policy: QueryPolicy::Default,
        }];
        let value = format!("https://gateway.test{suffix}?tenant=a&tenant=b");
        let endpoint = operation.resolve_legacy(&value, &query).unwrap();
        assert_eq!(endpoint.url().as_url().path(), path);
        assert_eq!(
            endpoint
                .url()
                .as_url()
                .query_pairs()
                .filter(|(key, _)| key == "api-version")
                .count(),
            1
        );
        assert_eq!(
            endpoint
                .url()
                .as_url()
                .query_pairs()
                .filter(|(key, _)| key == "tenant")
                .count(),
            2
        );
    }

    #[rstest]
    #[case::slash("a/b", "a%2Fb:analyze")]
    #[case::percent("a%2Fb", "a%252Fb:analyze")]
    #[case::unicode("é x", "%C3%A9%20x:analyze")]
    #[case::backslash("a\\b", "a%5Cb:analyze")]
    fn dynamic_action_preserves_prefix_and_query_policy(#[case] model: &str, #[case] action: &str) {
        let operation = AnalyzeDocument::new(model).unwrap();
        let target = EndpointTarget::Mount(
            ApiUrl::parse_mount(
                "https://gateway.test/team%2Fname?api-version=existing&tenant=a&tenant=b",
            )
            .unwrap(),
        );
        let resolved = operation
            .resolve_with_query(
                &target,
                &[QueryParameter {
                    key: "api-version",
                    value: "configured",
                    policy: QueryPolicy::Default,
                }],
            )
            .unwrap();
        assert_eq!(
            resolved.url().as_url().path(),
            format!("/team%2Fname/documentintelligence/documentModels/{action}")
        );
        assert_eq!(
            resolved.url().as_url().query(),
            Some("api-version=existing&tenant=a&tenant=b")
        );
        assert!(
            operation
                .matches(&format!("/documentintelligence/documentModels/{action}"))
                .unwrap()
        );
    }

    #[rstest]
    fn exact_destination_bypasses_generated_query() {
        let operation = AnalyzeDocument::new("model").unwrap();
        let url = ApiUrl::parse_exact("https://gateway.test/custom?sig=a%2fb+%20").unwrap();
        let resolved = operation
            .resolve_with_query(
                &EndpointTarget::Exact(url.clone()),
                &[QueryParameter {
                    key: "api-version",
                    value: "configured",
                    policy: QueryPolicy::Replace,
                }],
            )
            .unwrap();
        assert_eq!(resolved.url(), &url);
    }

    #[rstest]
    #[case::dot(".")]
    #[case::parent("..")]
    fn rejects_dot_identifiers_before_action_suffix(#[case] value: &str) {
        assert!(matches!(
            AnalyzeDocument::new(value),
            Err(ApiUrlError::InvalidSegment)
        ));
    }
}

pub fn resolve_messages(value: &str) -> Result<ResolvedEndpoint, ApiUrlError> {
    use crate::base_llm::endpoint::ProviderEndpoint;
    let exact = ApiUrl::<Complete>::parse_exact(value)?;
    let route = crate::anthropic::endpoints::AnthropicEndpoint::Messages.path();
    let suffix = format!("/{}", route.segments().collect::<Vec<_>>().join("/"));
    if exact
        .as_url()
        .path()
        .trim_end_matches('/')
        .ends_with(&suffix)
    {
        return Ok(ResolvedEndpoint::new(reqwest::Method::POST, exact));
    }
    let mut base = exact.into_url();
    let path = base.path().trim_end_matches('/');
    if let Some((prefix, _)) = path.split_once("/anthropic") {
        let prefix = prefix.to_owned();
        base.set_path(&prefix);
    }
    AzureEndpoint::Messages.resolve(&EndpointTarget::Mount(ApiUrl::parse_mount(base.as_str())?))
}
