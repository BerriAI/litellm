use crate::base_llm::endpoint::{ProviderEndpoint, ResolvedEndpoint};
use litellm_core_utils::{
    ApiUrlError,
    url_utils::{ApiUrl, Complete, EndpointTarget, Mount},
};

pub const DEFAULT_API_BASE: &str = "https://api.anthropic.com";

crate::provider_endpoints! {
    pub enum AnthropicEndpoint {
        Messages => POST "/v1/messages",
        CountTokens => POST "/v1/messages/count_tokens",
        CreateBatch => POST "/v1/messages/batches",
        ListBatches => GET "/v1/messages/batches",
        ListModels => GET "/v1/models",
        ListFiles => GET "/v1/files",
        ListSkills => GET "/v1/skills"
    }
}

pub fn resolve_messages(value: &str) -> Result<ResolvedEndpoint, ApiUrlError> {
    let mut parsed = ApiUrl::<Complete>::parse_exact(value)?.into_url();
    let path = parsed.path().trim_end_matches('/').to_owned();
    parsed.set_path(&path);
    let exact = ApiUrl::<Complete>::parse_exact(parsed.as_str())?;

    let target = if AnthropicEndpoint::Messages
        .path()
        .is_suffix_of(exact.as_url().path())
    {
        EndpointTarget::Exact(exact)
    } else {
        EndpointTarget::Mount(ApiUrl::<Mount>::parse_mount(value)?)
    };
    AnthropicEndpoint::Messages.resolve(&target)
}

#[derive(Clone, Debug)]
pub struct BatchId(String);

impl BatchId {
    pub fn new(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self(
            litellm_core_utils::url_utils::check_segment(value)?.to_owned(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self(
            litellm_core_utils::url_utils::check_segment(value)?.to_owned(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct FileId(String);

impl FileId {
    pub fn new(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self(
            litellm_core_utils::url_utils::check_segment(value)?.to_owned(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SkillId(String);

impl SkillId {
    pub fn new(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self(
            litellm_core_utils::url_utils::check_segment(value)?.to_owned(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SkillVersion(String);

impl SkillVersion {
    pub fn new(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self(
            litellm_core_utils::url_utils::check_segment(value)?.to_owned(),
        ))
    }
}

#[derive(Clone, Debug)]
pub enum AnthropicResourceEndpoint {
    RetrieveBatch(BatchId),
    CancelBatch(BatchId),
    DeleteBatch(BatchId),
    BatchResults(BatchId),
    RetrieveModel(ModelId),
    FileMetadata(FileId),
    DownloadFile(FileId),
    DeleteFile(FileId),
    RetrieveSkill(SkillId),
    DeleteSkill(SkillId),
    CreateSkillVersion(SkillId),
    ListSkillVersions(SkillId),
    RetrieveSkillVersion {
        skill: SkillId,
        version: SkillVersion,
    },
    DeleteSkillVersion {
        skill: SkillId,
        version: SkillVersion,
    },
}

impl ProviderEndpoint for AnthropicResourceEndpoint {
    fn resolve(&self, target: &EndpointTarget) -> Result<ResolvedEndpoint, ApiUrlError> {
        use AnthropicEndpoint::{ListBatches, ListFiles, ListModels, ListSkills};
        use reqwest::Method;

        let (method, base, id, suffix, version) = match self {
            Self::RetrieveBatch(id) => (Method::GET, ListBatches, &id.0, None, None),
            Self::CancelBatch(id) => (Method::POST, ListBatches, &id.0, Some("cancel"), None),
            Self::DeleteBatch(id) => (Method::DELETE, ListBatches, &id.0, None, None),
            Self::BatchResults(id) => (Method::GET, ListBatches, &id.0, Some("results"), None),
            Self::RetrieveModel(id) => (Method::GET, ListModels, &id.0, None, None),
            Self::FileMetadata(id) => (Method::GET, ListFiles, &id.0, None, None),
            Self::DownloadFile(id) => (Method::GET, ListFiles, &id.0, Some("content"), None),
            Self::DeleteFile(id) => (Method::DELETE, ListFiles, &id.0, None, None),
            Self::RetrieveSkill(id) => (Method::GET, ListSkills, &id.0, None, None),
            Self::DeleteSkill(id) => (Method::DELETE, ListSkills, &id.0, None, None),
            Self::CreateSkillVersion(id) => {
                (Method::POST, ListSkills, &id.0, Some("versions"), None)
            }
            Self::ListSkillVersions(id) => (Method::GET, ListSkills, &id.0, Some("versions"), None),
            Self::RetrieveSkillVersion { skill, version } => (
                Method::GET,
                ListSkills,
                &skill.0,
                Some("versions"),
                Some(version.0.as_str()),
            ),
            Self::DeleteSkillVersion { skill, version } => (
                Method::DELETE,
                ListSkills,
                &skill.0,
                Some("versions"),
                Some(version.0.as_str()),
            ),
        };
        let url = match target {
            EndpointTarget::Exact(url) => url.clone(),
            EndpointTarget::Mount(mount) => mount.resolve_segments(
                base.path()
                    .segments()
                    .chain([Some(id.as_str()), suffix, version].into_iter().flatten()),
                &[],
            )?,
        };
        Ok(ResolvedEndpoint::new(method, url))
    }
}

pub struct RetrieveBatch {
    pub id: BatchId,
}

impl ProviderEndpoint for RetrieveBatch {
    fn resolve(&self, target: &EndpointTarget) -> Result<ResolvedEndpoint, ApiUrlError> {
        AnthropicResourceEndpoint::RetrieveBatch(self.id.clone()).resolve(target)
    }
}

pub fn legacy_batch_mount(value: &str) -> Result<EndpointTarget, ApiUrlError> {
    let mut url = ApiUrl::<Complete>::parse_exact(value)?.into_url();
    let path = url.path().trim_end_matches('/');
    let batches = format!(
        "/{}",
        AnthropicEndpoint::CreateBatch
            .path()
            .segments()
            .collect::<Vec<_>>()
            .join("/")
    );
    let messages = format!(
        "/{}",
        AnthropicEndpoint::Messages
            .path()
            .segments()
            .collect::<Vec<_>>()
            .join("/")
    );
    let prefix = path
        .strip_suffix(&batches)
        .or_else(|| path.strip_suffix(&messages))
        .unwrap_or(path)
        .to_owned();
    url.set_path(&prefix);
    Ok(EndpointTarget::Mount(ApiUrl::parse_mount(url.as_str())?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::slash("batch/id", "batch%2Fid")]
    #[case::percent("batch%2Fid", "batch%252Fid")]
    #[case::space("batch id", "batch%20id")]
    fn retrieval_declares_get_and_keeps_identifiers_in_one_segment(
        #[case] value: &str,
        #[case] encoded: &str,
    ) {
        let operation = RetrieveBatch {
            id: BatchId::new(value).unwrap(),
        };
        let target =
            legacy_batch_mount("https://gateway.test/team%2Fname/v1/messages/batches?tenant=a")
                .unwrap();
        let endpoint = operation.resolve(&target).unwrap();
        assert_eq!(endpoint.method(), &reqwest::Method::GET);
        assert_eq!(
            endpoint.url().as_url().path(),
            format!("/team%2Fname/v1/messages/batches/{encoded}")
        );
        assert_eq!(endpoint.url().as_url().query(), Some("tenant=a"));
    }

    #[rstest]
    #[case::dot(".")]
    #[case::parent("..")]
    fn batch_identifiers_cannot_remove_route_segments(#[case] value: &str) {
        assert!(BatchId::new(value).is_err());
    }
}
