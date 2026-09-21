#[derive(Clone, Debug)]
pub struct ProviderIdentity {
    pub provider: String,
    pub requested_model: String,
    pub resolved_model: String,
    pub deployment: Option<String>,
    pub api_version: Option<String>,
    pub caller_dialect: crate::base_llm::base_model_iterator::ApiDialect,
    pub upstream_dialect: crate::base_llm::base_model_iterator::ApiDialect,
    pub delivery: crate::base_llm::base_model_iterator::DeliveryMode,
    pub operation: TranslationOperation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslationOperation {
    Create,
    Retrieve,
    Delete,
    ListInputItems,
    Cancel,
    Compact,
    CountTokens,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtensionPolicy {
    Preserve,
    Reject,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TranslationError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("provider error ({status_code}): {message}")]
    Provider {
        status_code: u16,
        message: String,
        headers: Box<[(String, String)]>,
    },
}

pub struct SignedRequest {
    _request: reqwest::Request,
}

impl SignedRequest {
    pub fn request(&self) -> &reqwest::Request {
        todo!()
    }
}

pub struct ProviderResponse<'a> {
    pub status: u16,
    pub headers: &'a [(String, String)],
    pub body: &'a [u8],
}

pub struct StreamLimits {
    pub max_buffer_bytes: usize,
    pub max_events_per_frame: usize,
}

pub enum StreamProgress<Event, Completion> {
    Events(Box<[Event]>),
    Finished {
        events: Box<[Event]>,
        completion: crate::base_llm::base_model_iterator::StreamOutcome<Completion>,
    },
}

pub trait SemanticDecoder: Sized {
    type Frame;
    type Event;
    type Completion;

    fn decode(
        &mut self,
        _frame: Self::Frame,
    ) -> Result<StreamProgress<Self::Event, Self::Completion>, TranslationError>;

    fn end_of_input(
        self,
    ) -> Result<
        crate::base_llm::base_model_iterator::StreamOutcome<Self::Completion>,
        TranslationError,
    >;
}

pub enum StreamTranslation<Decoder> {
    MatchingDialectRelay(crate::base_llm::base_model_iterator::MatchingDialectRelay),
    Translated(Decoder),
}

pub enum HookPosition {
    BeforeDocumentUpload,
    BeforeProviderSend,
}

pub enum PreparationEffect {
    FetchMedia,
    UploadDocument,
    SubmitJob,
    PollJob,
}

pub struct PreparationPolicy {
    pub effects: Box<[PreparationEffect]>,
    pub hook_position: HookPosition,
    pub max_repair_attempts: usize,
    pub request_timeout: Option<std::time::Duration>,
    pub total_deadline: Option<std::time::Instant>,
}

pub trait AttemptPreparation: Send + Sync {
    type Input;
    type Prepared;
    type Environment;

    fn policy(&self) -> PreparationPolicy;

    fn prepare(
        &self,
        _input: &Self::Input,
        _environment: &Self::Environment,
    ) -> impl std::future::Future<Output = Result<Self::Prepared, TranslationError>> + Send;
}
