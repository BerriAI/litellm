//! Nace document parsing as an OCR provider.
//!
//! A Nace parse is an async job. The create call waits up to `wait_seconds` and the
//! adapter polls the job against the API base until it ends. Once the result is read, or
//! the call fails after the job exists, the adapter deletes the job: a job that is still
//! running is cancelled, which Nace does not bill, and a finished job has its stored result
//! erased. A document Nace cannot download by link is uploaded through a one-time upload
//! grant first, and the uploaded file is deleted when the call ends, whatever its outcome.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::RangeInclusive,
    sync::LazyLock,
    time::Duration,
};

use fancy_regex::Regex;

use litellm_auth::InputSource;
use litellm_core_utils::{
    call_arguments::{CallArguments, compose_body},
    url_utils::{ApiUrl, Complete},
};
use litellm_http::request::{HeaderPolicy, has_header, http_request, with_headers};
use litellm_llms_types::formats::ocr::{
    LiteLLMOcrResponse, OcrDocument, OcrPage, OcrResponseFormat, OcrUsageInfo,
};
use litellm_secret_value::SecretValue;
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use serde_with::{DefaultOnNull, serde_as};
use tokio::time::Instant;

use crate::{
    base_llm::ocr::{
        document::{InlineDocument, inline_remote_document},
        error::Error,
        handler::{CallHooks, OcrClient, read_response_bytes},
        transformation::{
            BaseOcrConfig, DROP_PARAMS_PARAM, DecodedOcrResponse, OCR_INLINE_MAX_BYTES,
            OCR_POLL_RETRY_SECS, OcrConnection, OcrRequestContext, OcrResponseContext,
            PreparedOcrRequest, decode_and_normalize_response, decode_response,
            drops_unsupported_params,
        },
    },
    reducto::ocr::transformation::prepare_upload_request,
};

const PROVIDER: &str = "Nace";
const NACE_API_BASE: &str = "https://console.nace.ai";
const NACE_API_KEY_ENV: &str = "NACE_API_KEY";
const NACE_API_BASE_ENVS: [&str; 2] = ["NACE_API_BASE", "NACE_BASE_URL"];
const MODELS: &[&str] = &["parse-low", "parse-medium", "parse-high"];
const MIN_WAIT_SECONDS: u64 = 1;
const MAX_WAIT_SECONDS: u64 = 60;
const UPLOAD_GRANT_TTL_SECONDS: u64 = 600;
const UPLOAD_PATH_PREFIX: &str = "litellm";
const UPLOAD_TOKEN_HEADER: &str = "X-Upload-Token";
/// Cleanup runs after the caller's timeout may have run out, so each DELETE has its own
/// timeout, at most 10 seconds for the job and the uploaded file of one call.
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_USER_AGENT: &str = "litellm";
const OUTPUT_PARAM: &str = "output";
const INCLUDE_IMAGE_BASE64_PARAM: &str = "include_image_base64";
const INCLUDE_IMAGES_OUTPUT: &str = "include_images";
const PAGE_RANGES_PARAM: &str = "page_ranges";
const PAGES_PARAM: &str = "pages";
const RESERVED_PARAMS: &[&str] = &["source", "corpus_sources", "parse_mode", "parser"];
const SUPPORTED_PARAMS: &[&str] = &[
    PAGE_RANGES_PARAM,
    PAGES_PARAM,
    "figures",
    "diagrams",
    "ocr",
    "spreadsheet",
    "password",
    "chunking",
    "quality",
    "name",
    "project_id",
    OUTPUT_PARAM,
    INCLUDE_IMAGE_BASE64_PARAM,
    DROP_PARAMS_PARAM,
];
const IMAGES_REASON: &str =
    "Nace returns images only as links that need the Nace API key and are deleted with the job";
const URL_SOURCE_EXTENSIONS: &[&str] = &[
    "avi", "csv", "doc", "docm", "docx", "eml", "flac", "htm", "html", "jpeg", "jpg", "json",
    "log", "m4a", "md", "mdx", "mkv", "mov", "mp3", "mp4", "mpga", "msg", "oft", "oga", "ogg",
    "parquet", "pdf", "png", "ppt", "pptx", "tif", "tiff", "txt", "vsd", "vsdx", "wav", "webp",
    "xls", "xlsm", "xlsx", "xml",
];
const CONTENT_TRUNCATED: &str = "content_truncated";
const DOCUMENT_CONTENT: &str = "document_content";
const NOT_RETAINED: &str = "not_retained";
const WORKSPACE_FILE: &str = "workspace_file";
/// The most pages one response is split into, whatever the job reports.
const MAX_PAGES: usize = 10_000;
/// The most characters of Nace error text passed on to the caller.
const MAX_MESSAGE_CHARS: usize = 500;

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    serde_with::SerializeDisplay,
)]
pub enum ParseMode {
    #[strum(serialize = "low")]
    Low,
    #[strum(serialize = "medium")]
    Medium,
    #[strum(serialize = "high")]
    High,
}

#[derive(Clone, Debug)]
pub struct NaceParseParams {
    parse_mode: ParseMode,
    output: Map<String, Value>,
    options: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum NaceSource {
    #[serde(rename = "url")]
    Url { url: String, file_name: String },
    #[serde(rename = "workspace_file")]
    WorkspaceFile {
        workspace_id: String,
        file_id: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct NaceParseRequest {
    source: NaceSource,
    parse_mode: ParseMode,
    output: Map<String, Value>,
    #[serde(flatten)]
    options: Map<String, Value>,
}

#[derive(Debug, thiserror::Error)]
enum Refusal {
    #[error("`{0}` is set by LiteLLM from the model and the document")]
    ReservedParam(String),
    #[error("`output` must be an object")]
    OutputShape,
    #[error(
        "`page_ranges` must be a list of {{\"start\", \"end\"}} integers with 1 <= start <= end"
    )]
    PageRanges,
    #[error("`pages` must be a list of non-negative integers")]
    Pages,
    #[error("pass either `pages` or `page_ranges`, not both")]
    PagesConflict,
    #[error(
        "pass `password` as the top-level option, not in `extra_body`, so that the logging callbacks mask it"
    )]
    NestedPassword,
    #[error(
        "the upload URL must be on the API base origin, or https on a public address, without credentials"
    )]
    UploadUrl,
    #[error("the job read is not the job this call created")]
    OtherJob,
    #[error("the job has no id")]
    NoJobId,
    #[error(
        "the content URL must be under this job's path on the API base origin, without credentials"
    )]
    ContentUrl,
}

#[derive(Serialize)]
struct UploadGrantRequest<'a> {
    path: &'a str,
    ttl_seconds: u64,
}

#[derive(Deserialize)]
struct UploadGrant {
    upload_url: String,
    token: SecretValue,
}

#[derive(Deserialize)]
struct UploadJob {
    result: UploadResult,
}

#[derive(Deserialize)]
struct UploadResult {
    file: UploadedFile,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
struct UploadedFile {
    workspace_id: String,
    file_id: String,
}

#[derive(
    Clone, Debug, PartialEq, strum::EnumString, strum::Display, serde_with::DeserializeFromStr,
)]
enum JobStatus {
    #[strum(serialize = "queued")]
    Queued,
    #[strum(serialize = "running")]
    Running,
    #[strum(serialize = "succeeded")]
    Succeeded,
    #[strum(serialize = "failed")]
    Failed,
    #[strum(serialize = "cancelled")]
    Cancelled,
    #[strum(default, transparent)]
    Unknown(String),
}

/// One read of a Nace document job, keeping only what the adapter acts on.
#[derive(Debug, Deserialize)]
pub struct ParseJob {
    job_id: String,
    status: JobStatus,
    units: Option<i64>,
    result: Option<ParseResult>,
    result_state: Option<String>,
    error: Option<JobError>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct ParseResult {
    document: ParsedDocument,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    #[serde(default)]
    disclosures: Vec<Disclosure>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct ParsedDocument {
    markdown: Option<String>,
    content_url: Option<String>,
    page_count: Option<i64>,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    #[serde(default)]
    blocks: Vec<Map<String, Value>>,
}

#[derive(Debug, Deserialize)]
struct Disclosure {
    code: String,
    target: Option<String>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct JobError {
    #[serde_as(deserialize_as = "DefaultOnNull")]
    #[serde(default)]
    code: String,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    #[serde(default)]
    message: String,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    #[serde(default)]
    retryable: bool,
}

/// One entry of the request's `page_ranges`, 1-based and inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct PageRange {
    start: i64,
    end: i64,
}

#[derive(Clone, Debug)]
pub struct NaceParseConfig;

impl BaseOcrConfig for NaceParseConfig {
    type OcrParams = NaceParseParams;
    type ProviderRequest = NaceParseRequest;
    type Environment = Vec<(String, String)>;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        SUPPORTED_PARAMS
    }

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(NACE_API_KEY_ENV)
    }

    fn secret_names(&self) -> Vec<&'static str> {
        std::iter::once(NACE_API_KEY_ENV)
            .chain(NACE_API_BASE_ENVS)
            .collect()
    }

    /// Everything that can refuse the request is checked here, before the document is
    /// uploaded, so a refused request never leaves a file behind. That includes the shape of
    /// `extra_body`, which the body composed after the upload merges in.
    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        model: &str,
    ) -> Result<NaceParseParams, Error> {
        let parse_mode = parse_mode(model)?;
        if let Some(name) = reserved_param(non_default_params) {
            return Err(Error::request_field(
                name.clone(),
                Refusal::ReservedParam(name),
            ));
        }
        if non_default_params
            .get("extra_body")
            .and_then(|body| body.get("password"))
            .is_some()
        {
            return Err(Error::request_field(
                "extra_body.password",
                Refusal::NestedPassword,
            ));
        }
        if non_default_params.get(INCLUDE_IMAGE_BASE64_PARAM) == Some(&Value::Bool(true))
            && !drops_unsupported_params(non_default_params)
        {
            return Err(Error::UnsupportedParam {
                param: INCLUDE_IMAGE_BASE64_PARAM.to_owned(),
                reason: IMAGES_REASON,
            });
        }
        compose_body(non_default_params, &Map::new(), SUPPORTED_PARAMS)?;
        requested_page_ranges(non_default_params)?;
        let output = match non_default_params.get(OUTPUT_PARAM) {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(output)) => output.clone(),
            Some(_) => return Err(Error::request_field(OUTPUT_PARAM, Refusal::OutputShape)),
        };
        if output.get(INCLUDE_IMAGES_OUTPUT) == Some(&Value::Bool(true))
            && !drops_unsupported_params(non_default_params)
        {
            return Err(Error::UnsupportedParam {
                param: format!("{OUTPUT_PARAM}.{INCLUDE_IMAGES_OUTPUT}"),
                reason: IMAGES_REASON,
            });
        }
        let mapped_pages = pages_to_ranges(non_default_params.get(PAGES_PARAM))?;
        Ok(NaceParseParams {
            parse_mode,
            output: page_split_output(
                output
                    .into_iter()
                    .filter(|(name, value)| {
                        !(name == INCLUDE_IMAGES_OUTPUT && *value == Value::Bool(true))
                    })
                    .collect(),
            ),
            options: non_default_params
                .select(SUPPORTED_PARAMS)
                .into_iter()
                .filter(|(name, _)| {
                    !matches!(
                        name.as_str(),
                        OUTPUT_PARAM | INCLUDE_IMAGE_BASE64_PARAM | PAGES_PARAM | DROP_PARAMS_PARAM
                    )
                })
                .chain(mapped_pages.map(|ranges| (PAGE_RANGES_PARAM.to_owned(), json!(ranges))))
                .collect(),
        })
    }

    /// The code and message of the Nace error envelope (`api_error_text`).
    fn get_error_class(
        &self,
        error_message: String,
        status_code: u16,
        headers: reqwest::header::HeaderMap,
    ) -> Error {
        Error::Provider {
            status: status_code,
            body: api_error_text(&error_message, &[]),
            headers: Box::new(headers),
        }
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        resolve_headers(&request.connection)
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, Error> {
        let wait_seconds = wait_seconds(request.connection.timeout).to_string();
        Ok(api_url(&request.connection, &["v1", "documents", "parse"])?
            .append_query_pairs([("wait_seconds", wait_seconds.as_str())])
            .into_string())
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        _headers: &[(String, String)],
    ) -> Result<Self::ProviderRequest, Error> {
        let source = url_source(&document).ok_or(Error::UnsupportedDocument {
            provider: PROVIDER,
            guidance: "only an https link with a file extension is sent without an upload, use the async OCR route for other documents",
        })?;
        Ok(optional_params.request(source))
    }

    async fn async_transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<Self::ProviderRequest, Error> {
        let source = match url_source(&document) {
            Some(source) => source,
            None => upload_document(document, headers, context).await?,
        };
        Ok(optional_params.request(source))
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error> {
        decode_and_normalize_response(model, raw_response, request_format, |model, job| {
            normalize_job(model, job, None, &[])
        })
    }

    async fn async_transform_ocr_response(
        &self,
        model: &str,
        raw_response: reqwest::Response,
        context: OcrResponseContext<'_>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        let bytes = read_bytes(
            raw_response,
            context.connection.max_response_bytes,
            context.headers,
        )
        .await?;
        // The job id is read before anything that can fail, so the job is deleted even when the
        // rest of the body does not decode or the response hook rejects it.
        let job_id = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|job| job.get("job_id")?.as_str().map(str::to_owned))
            .filter(|job_id| !job_id.trim().is_empty());
        let followups = Followups::new(&context);
        let response = match context
            .hooks
            .response_received(&without_source(&bytes, context.headers))
            .await
        {
            Ok(()) => match decode_response::<ParseJob>(&bytes, followups.native) {
                Ok(created) => followups.complete(model, created).await,
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };
        if let Some(job_id) = job_id {
            delete_job(context.client, context.connection, context.headers, &job_id).await;
        }
        response
    }

    /// The Reducto flow: guardrails see the source document before it is uploaded. Every
    /// step after the upload reuses headers and arguments that the grant request and
    /// `map_ocr_params` already accepted, so it cannot fail and leave the file behind.
    async fn prepare_request(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
        hooks: &dyn CallHooks<Error>,
    ) -> Result<reqwest::Request, Error> {
        prepare_upload_request(self, request, client, hooks).await
    }

    fn request_cleanup(&self, request: &reqwest::Request) -> Option<Value> {
        let body =
            serde_json::from_slice::<Map<String, Value>>(request.body()?.as_bytes()?).ok()?;
        body.get("source")
            .filter(|source| source.get("type").and_then(Value::as_str) == Some(WORKSPACE_FILE))
            .cloned()
    }

    async fn release_request(
        &self,
        cleanup: Value,
        context: OcrRequestContext<'_>,
        headers: &[(String, String)],
    ) {
        if let Ok(file) = serde_json::from_value::<UploadedFile>(cleanup) {
            delete_file(context.client, context.connection, headers, &file).await;
        }
    }
}

impl NaceParseParams {
    fn request(&self, source: NaceSource) -> NaceParseRequest {
        NaceParseRequest {
            source,
            parse_mode: self.parse_mode,
            output: self.output.clone(),
            options: self.options.clone(),
        }
    }
}

impl ParseJob {
    fn is_pending(&self) -> bool {
        matches!(self.status, JobStatus::Queued | JobStatus::Running)
    }
}

fn parse_mode(model: &str) -> Result<ParseMode, Error> {
    model
        .strip_prefix("parse-")
        .and_then(|mode| mode.parse().ok())
        .ok_or_else(|| Error::InvalidModel {
            provider: "nace",
            model: model.to_owned(),
            supported: MODELS,
        })
}

fn reserved_param(arguments: &CallArguments) -> Option<String> {
    let extra_body = arguments
        .get("extra_body")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|fields| fields.keys())
        .filter(|name| name.as_str() == OUTPUT_PARAM || RESERVED_PARAMS.contains(&name.as_str()));
    arguments
        .keys()
        .filter(|name| RESERVED_PARAMS.contains(&name.as_str()))
        .chain(extra_body)
        .next()
        .cloned()
}

fn page_split_output(output: Map<String, Value>) -> Map<String, Value> {
    let formats = match output.get("formats") {
        Some(Value::Array(formats)) if !formats.contains(&json!("markdown")) => Some(Value::Array(
            formats
                .iter()
                .cloned()
                .chain(std::iter::once(json!("markdown")))
                .collect(),
        )),
        _ => None,
    };
    output
        .into_iter()
        .chain(formats.map(|formats| ("formats".to_owned(), formats)))
        .chain(std::iter::once((
            "include_page_markers".to_owned(),
            Value::Bool(true),
        )))
        .collect()
}

/// Half the request timeout, from 1 to 60 seconds. A zero retention account refuses a create
/// without a wait, and keeps a result only when the job ends inside it.
fn wait_seconds(timeout: Duration) -> u64 {
    (timeout.as_secs() / 2).clamp(MIN_WAIT_SECONDS, MAX_WAIT_SECONDS)
}

fn api_base(connection: &OcrConnection) -> String {
    connection
        .api_base
        .iter()
        .cloned()
        .chain(
            NACE_API_BASE_ENVS
                .iter()
                .filter_map(|name| connection.secret(name)),
        )
        .map(|base| base.trim().to_owned())
        .find(|base| !base.is_empty())
        .unwrap_or_else(|| NACE_API_BASE.to_owned())
}

fn api_url(connection: &OcrConnection, path: &[&str]) -> Result<ApiUrl<Complete>, Error> {
    ApiUrl::parse(&api_base(connection))
        .and_then(|url| url.complete_path(path))
        .map_err(|error| Error::request_field("api_base", error))
}

fn api_origin(connection: &OcrConnection) -> Result<url::Origin, Error> {
    Url::parse(&api_base(connection))
        .map(|url| url.origin())
        .map_err(|error| Error::request_field("api_base", error))
}

fn resolve_headers(connection: &OcrConnection) -> Result<Vec<(String, String)>, Error> {
    if has_header(&connection.extra_headers, "authorization") {
        validate_destination(connection, connection.extra_headers_source)?;
        return Ok(connection.extra_headers.clone());
    }
    let (api_key, source) = connection
        .api_key
        .as_ref()
        .map(|key| key.expose().trim().to_owned())
        .filter(|key| !key.is_empty())
        .map(|key| (key, connection.api_key_source))
        .or_else(|| {
            connection
                .secret(NACE_API_KEY_ENV)
                .map(|key| key.trim().to_owned())
                .filter(|key| !key.is_empty())
                .map(|key| (key, InputSource::Environment))
        })
        .ok_or(Error::MissingCredential {
            credential: NACE_API_KEY_ENV,
            guidance: "set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()",
        })?;
    validate_destination(connection, source)?;
    Ok(
        std::iter::once(("Authorization".to_owned(), format!("Bearer {api_key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

fn validate_destination(
    connection: &OcrConnection,
    credential_source: InputSource,
) -> Result<(), Error> {
    if connection.api_base.is_some()
        && connection.api_base_source == InputSource::Request
        && credential_source != InputSource::Request
    {
        return Err(litellm_auth::Error::InvalidConfiguration(
            litellm_auth::ErrorDetail::RequestControlledEndpoint { provider: PROVIDER },
        )
        .into());
    }
    Ok(())
}

fn url_source(document: &OcrDocument) -> Option<NaceSource> {
    let url = Url::parse(document.source())
        .ok()
        .filter(|url| url.scheme() == "https")?;
    let file_name = document_name(document)
        .or_else(|| url.path_segments()?.next_back())
        .filter(|name| has_url_source_extension(name))?;
    Some(NaceSource::Url {
        url: document.source().to_owned(),
        file_name: file_name.to_owned(),
    })
}

fn document_name(document: &OcrDocument) -> Option<&str> {
    let (OcrDocument::DocumentUrl { extra_fields, .. }
    | OcrDocument::ImageUrl { extra_fields, .. }) = document;
    extra_fields.get("document_name")?.as_deref()
}

fn has_url_source_extension(file_name: &str) -> bool {
    file_name.rsplit_once('.').is_some_and(|(_, extension)| {
        URL_SOURCE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
    })
}

fn upload_extension(mime_type: &str) -> Option<&'static str> {
    Some(match mime_type {
        "application/pdf" => "pdf",
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/tiff" => "tiff",
        "image/webp" => "webp",
        "text/html" => "html",
        "text/plain" => "txt",
        "text/markdown" => "md",
        "text/csv" => "csv",
        "application/json" => "json",
        "message/rfc822" => "eml",
        "application/msword" => "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
        "application/vnd.ms-excel" => "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
        "application/vnd.ms-powerpoint" => "ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => "pptx",
        _ => return None,
    })
}

fn upload_path(mime_type: &str) -> String {
    let name = uuid::Uuid::new_v4().simple().to_string();
    match upload_extension(mime_type) {
        Some(extension) => format!("{UPLOAD_PATH_PREFIX}/{name}.{extension}"),
        None => format!("{UPLOAD_PATH_PREFIX}/{name}"),
    }
}

async fn upload_document(
    document: OcrDocument,
    headers: &[(String, String)],
    context: OcrRequestContext<'_>,
) -> Result<NaceSource, Error> {
    let OcrRequestContext { client, connection } = context;
    let document = match document.is_remote() {
        true => inline_remote_document(client.document_fetcher(), document, connection).await?,
        false => document,
    };
    let inline = InlineDocument::parse(document.source())?.ok_or(Error::UnsupportedDocument {
        provider: PROVIDER,
        guidance: "pass an https or http link, a data URI, or a file",
    })?;
    let mime_type = format!(
        "{}/{}",
        inline.mime_type().type_,
        inline.mime_type().subtype
    );
    let bytes = inline.decode(OCR_INLINE_MAX_BYTES)?;
    let size = bytes.len();
    let path = upload_path(&mime_type);
    let grant_request = litellm_http::outbound::json(
        &api_url(connection, &["v1", "documents", "upload-grants"])?.into_string(),
        headers,
        &UploadGrantRequest {
            path: &path,
            ttl_seconds: UPLOAD_GRANT_TTL_SECONDS,
        },
        Some(connection.timeout),
    )?;
    let grant_response = client
        .polling_http()
        .execute(grant_request)
        .await
        .map_err(litellm_http::Error::from)?;
    let grant = decode_response::<UploadGrant>(
        &read_bytes(grant_response, connection.max_response_bytes, headers).await?,
        false,
    )?
    .data;
    let (upload_url, route) = upload_target(&grant.upload_url, connection)?;
    let file_name = path.rsplit('/').next().unwrap_or(&path).to_owned();
    let file = reqwest::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str(&mime_type)
        .map_err(|_| Error::InvalidMimeType(mime_type.clone()))?;
    let form = reqwest::multipart::Form::new().part("file", file).text(
        "metadata",
        json!({ "path": path, "total_size_bytes": size }).to_string(),
    );
    let upload = client
        .polling_http()
        .post(upload_url)
        .header(UPLOAD_TOKEN_HEADER, grant.token.expose())
        // The upload host refuses a request without a User-Agent. The operator's configured
        // one is the clients' default, so it is sent as is, and `litellm` only stands in for none.
        .header(
            reqwest::header::USER_AGENT,
            client.user_agent().unwrap_or(DEFAULT_USER_AGENT),
        )
        .multipart(form)
        .timeout(connection.timeout)
        .build()
        .map_err(|error| litellm_http::Error::from(error.without_url()))?;
    // The upload URL can carry a signature, so transport errors leave it out.
    let response = match route {
        UploadRoute::ApiBase => client
            .polling_http()
            .execute(upload)
            .await
            .map_err(|error| litellm_http::Error::from(error.without_url()))?,
        UploadRoute::Checked => client
            .document_fetcher()
            .execute_checked(upload)
            .await
            .map_err(|error| match error.is_request() {
                true => Error::response_field("upload_url", Refusal::UploadUrl),
                false => error.into(),
            })?,
    };
    // Only the status of a refused upload reaches the caller, and a body that does not decode
    // gives a fixed text, since the upload host can echo the token it was sent.
    if !response.status().is_success() {
        return Err(Error::Provider {
            status: response.status().as_u16(),
            body: "the upload host refused the document".into(),
            headers: Box::default(),
        });
    }
    let body = read_response_bytes(response, connection.max_response_bytes).await?;
    let uploaded = serde_json::from_slice::<UploadJob>(&body)
        .map_err(|_| Error::ResponseField {
            path: "result.file".into(),
            cause: None,
        })?
        .result
        .file;
    Ok(NaceSource::WorkspaceFile {
        workspace_id: uploaded.workspace_id,
        file_id: uploaded.file_id,
    })
}

enum UploadRoute {
    /// The operator's own endpoint, sent with the polling client.
    ApiBase,
    /// Any other host, sent only through the gateway's document URL policy.
    Checked,
}

/// Where the grant's upload URL may be sent. The API base origin is the operator's own
/// endpoint. Anywhere else must be https, and the request is sent through the URL policy the
/// gateway applies to caller-supplied document links (`MediaFetcher::execute_checked`).
fn upload_target(raw: &str, connection: &OcrConnection) -> Result<(Url, UploadRoute), Error> {
    let refused = || Error::response_field("upload_url", Refusal::UploadUrl);
    let url = Url::parse(raw).map_err(|error| Error::response_field("upload_url", error))?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err(refused());
    }
    if url.origin() == api_origin(connection)? {
        return Ok((url, UploadRoute::ApiBase));
    }
    match url.scheme() {
        "https" => Ok((url, UploadRoute::Checked)),
        _ => Err(refused()),
    }
}

/// `read_response_bytes` with the body of an error response reduced to `api_error_text`, with
/// the API key in `headers` masked.
async fn read_bytes(
    response: reqwest::Response,
    limit: usize,
    headers: &[(String, String)],
) -> Result<bytes::Bytes, Error> {
    read_response_bytes(response, limit)
        .await
        .map_err(|error| match error {
            Error::Provider {
                status,
                body,
                headers: response_headers,
            } => Error::Provider {
                status,
                body: api_error_text(&body, &api_key(headers).into_iter().collect::<Vec<_>>()),
                headers: response_headers,
            },
            other => other,
        })
}

fn api_key(headers: &[(String, String)]) -> Option<&str> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.strip_prefix("Bearer ").unwrap_or(value).trim())
}

/// A job as the response hooks see it: without its `source`, which can be the caller's signed
/// link, and with `error.code` and `error.message` cleaned by `clean_text`, which also masks the
/// API key in `headers`. Other OCR providers do not echo the document link in their responses
/// either.
fn without_source(job: &[u8], headers: &[(String, String)]) -> Vec<u8> {
    let secrets: Vec<&str> = api_key(headers).into_iter().collect();
    serde_json::from_slice::<Map<String, Value>>(job)
        .ok()
        .and_then(|mut job| {
            job.shift_remove("source");
            if let Some(Value::Object(error)) = job.get_mut("error") {
                for field in ["code", "message"] {
                    if let Some(Value::String(text)) = error.get_mut(field) {
                        *text = clean_text(text, &secrets);
                    }
                }
            }
            serde_json::to_vec(&job).ok()
        })
        .unwrap_or_else(|| job.to_vec())
}

/// Calls that follow the parse create: job polls and the full-content download. Both go to the
/// API base with only the `Authorization` header of the create call.
struct Followups<'a> {
    client: &'a OcrClient,
    connection: &'a OcrConnection,
    hooks: &'a dyn CallHooks<Error>,
    headers: &'a [(String, String)],
    native: bool,
    page_ranges: Vec<PageRange>,
}

impl<'a> Followups<'a> {
    fn new(context: &OcrResponseContext<'a>) -> Self {
        Self {
            client: context.client,
            connection: context.connection,
            hooks: context.hooks,
            headers: context.headers,
            native: context.request_format == OcrResponseFormat::Native,
            page_ranges: requested_page_ranges(context.optional_params).unwrap_or_default(),
        }
    }

    async fn complete(
        &self,
        model: &str,
        created: DecodedOcrResponse<ParseJob>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        if created.data.job_id.trim().is_empty() {
            return Err(Error::response_field("job_id", Refusal::NoJobId));
        }
        let deadline = Instant::now()
            .checked_add(self.connection.settings.poll_timeout)
            .ok_or(Error::PollTimeout)?;
        let finished = match created.data.is_pending() {
            true => self.poll(&created.data.job_id, deadline).await?,
            false => created,
        };
        let full_markdown = match truncated_content(&finished.data)? {
            Some(content_url) => Some(
                self.fetch_content(content_url, &finished.data.job_id, deadline)
                    .await?,
            ),
            None => None,
        };
        // The content link dies with the job, so a native response carries the full markdown
        // in place of the preview, and no link.
        let mut native = finished.native;
        if let (Some(markdown), Some(document)) = (
            &full_markdown,
            native
                .as_mut()
                .and_then(|job| job.get_mut("result")?.get_mut("document")?.as_object_mut()),
        ) {
            document.insert("markdown".to_owned(), Value::String(markdown.clone()));
            document.remove("content_url");
        }
        Ok(LiteLLMOcrResponse {
            provider_native_response: native,
            ..normalize_job(model, finished.data, full_markdown, &self.page_ranges)?
        })
    }

    async fn poll(
        &self,
        job_id: &str,
        deadline: Instant,
    ) -> Result<DecodedOcrResponse<ParseJob>, Error> {
        let url = api_url(self.connection, &["v1", "documents", "jobs", job_id])?.into_string();
        let mut retry = OCR_POLL_RETRY_SECS;
        loop {
            sleep_until_retry(retry, deadline).await?;
            let response = self.get(&url, deadline).await?;
            retry = retry_after(&response);
            let decoded = tokio::time::timeout_at(
                deadline,
                read_bytes(response, self.connection.max_response_bytes, self.headers),
            )
            .await
            .map_err(|_| Error::PollTimeout)??;
            let job = decode_response::<ParseJob>(&decoded, self.native)?;
            if job.data.job_id != job_id {
                return Err(Error::response_field("job_id", Refusal::OtherJob));
            }
            if !job.data.is_pending() {
                self.hooks
                    .response_received(&without_source(&decoded, self.headers))
                    .await?;
                return Ok(job);
            }
        }
    }

    /// Reads the full markdown of a truncated result. The link is one of the job's own files,
    /// which Nace serves only with the API key, so it must sit under this job's path on the
    /// API base origin before the key goes with it.
    async fn fetch_content(
        &self,
        content_url: &str,
        job_id: &str,
        deadline: Instant,
    ) -> Result<String, Error> {
        let url = job_file_url(self.connection, job_id, content_url)?;
        loop {
            let response = self.get(url.as_str(), deadline).await?;
            if response.status() != StatusCode::ACCEPTED {
                let bytes = tokio::time::timeout_at(
                    deadline,
                    read_bytes(response, self.connection.max_response_bytes, self.headers),
                )
                .await
                .map_err(|_| Error::PollTimeout)??;
                return String::from_utf8(bytes.to_vec())
                    .map_err(|error| Error::response_field("result.document.content_url", error));
            }
            sleep_until_retry(retry_after(&response), deadline).await?;
        }
    }

    async fn get(&self, url: &str, deadline: Instant) -> Result<reqwest::Response, Error> {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(Error::PollTimeout)?;
        let builder = self
            .client
            .polling_http()
            .get(url)
            .timeout(remaining.min(self.connection.timeout));
        let builder = with_headers(
            builder,
            self.headers,
            HeaderPolicy::Only(&["authorization"]),
        );
        Ok(tokio::time::timeout_at(deadline, http_request(builder))
            .await
            .map_err(|_| Error::PollTimeout)??)
    }
}

/// A link to one of the job's own files: on the API base origin, without credentials, with
/// path segments that continue the API base path with `v1/documents/jobs/{job_id}` and at least
/// one more segment, and without an encoded `/`, `\` or `%` that could change the path later.
/// Only such a link receives the API key.
fn job_file_url(connection: &OcrConnection, job_id: &str, raw: &str) -> Result<Url, Error> {
    let field = "result.document.content_url";
    if job_id.trim().is_empty() {
        return Err(Error::response_field(field, Refusal::NoJobId));
    }
    let url = Url::parse(raw).map_err(|error| Error::response_field(field, error))?;
    let job = Url::parse(&api_url(connection, &["v1", "documents", "jobs", job_id])?.into_string())
        .map_err(|error| Error::response_field(field, error))?;
    let segments = |url: &Url| -> Vec<String> {
        url.path_segments()
            .into_iter()
            .flatten()
            .filter(|segment| !segment.is_empty())
            .map(str::to_owned)
            .collect()
    };
    let (job_path, path) = (segments(&job), segments(&url));
    let encoded = url.path().to_ascii_lowercase();
    match url.origin() == job.origin()
        && url.username().is_empty()
        && url.password().is_none()
        && !["%2f", "%5c", "%25"]
            .iter()
            .any(|code| encoded.contains(code))
        && path.len() > job_path.len()
        && path.starts_with(&job_path)
    {
        true => Ok(url),
        false => Err(Error::response_field(field, Refusal::ContentUrl)),
    }
}

/// Deletes the job, best effort. Nace cancels a job that is still running, without a charge,
/// and erases the stored result of a finished one, which keeps its charge.
async fn delete_job(
    client: &OcrClient,
    connection: &OcrConnection,
    headers: &[(String, String)],
    job_id: &str,
) {
    let path = ["v1", "documents", "jobs", job_id];
    if let Err(error) = delete(client, connection, headers, &path).await {
        let status = error.http_status_code();
        tracing::warn!(
            "Nace OCR could not delete its parse job (status {status:?}), a running job keeps running and is billed"
        );
    }
}

async fn delete_file(
    client: &OcrClient,
    connection: &OcrConnection,
    headers: &[(String, String)],
    file: &UploadedFile,
) {
    let path = [
        "v1",
        "documents",
        "workspaces",
        &file.workspace_id,
        "files",
        &file.file_id,
    ];
    if let Err(error) = delete(client, connection, headers, &path).await {
        let status = error.http_status_code();
        tracing::warn!(
            "Nace OCR could not delete the file it uploaded for parsing (status {status:?}), it stays in the Nace workspace"
        );
    }
}

async fn delete(
    client: &OcrClient,
    connection: &OcrConnection,
    headers: &[(String, String)],
    path: &[&str],
) -> Result<(), Error> {
    let builder = client
        .polling_http()
        .delete(api_url(connection, path)?.into_string())
        .timeout(CLEANUP_TIMEOUT);
    let builder = with_headers(builder, headers, HeaderPolicy::Only(&["authorization"]));
    read_bytes(
        http_request(builder).await?,
        connection.max_response_bytes,
        headers,
    )
    .await?;
    Ok(())
}

async fn sleep_until_retry(seconds: u64, deadline: Instant) -> Result<(), Error> {
    tokio::time::timeout_at(
        deadline,
        tokio::time::sleep(Duration::from_secs(seconds.max(1))),
    )
    .await
    .map_err(|_| Error::PollTimeout)
}

fn retry_after(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(OCR_POLL_RETRY_SECS)
}

fn truncated_content(job: &ParseJob) -> Result<Option<&str>, Error> {
    let Some(result) = job
        .result
        .as_ref()
        .filter(|_| job.status == JobStatus::Succeeded)
    else {
        return Ok(None);
    };
    let mut truncated = result
        .disclosures
        .iter()
        .filter(|disclosure| disclosure.code == CONTENT_TRUNCATED)
        .peekable();
    if truncated.peek().is_none() {
        return Ok(None);
    }
    if truncated.any(|disclosure| disclosure.target.as_deref() != Some(DOCUMENT_CONTENT)) {
        return Err(Error::RejectedDocument {
            provider: PROVIDER,
            message: "the workbook result is over 1 MiB, so Nace returned only sheet previews"
                .into(),
            guidance: "parse fewer sheets with the spreadsheet options",
        });
    }
    result
        .document
        .content_url
        .as_deref()
        .map(Some)
        .ok_or(Error::ResponseField {
            path: "result.document.content_url".into(),
            cause: None,
        })
}

fn requested_page_ranges(params: &CallArguments) -> Result<Vec<PageRange>, Error> {
    let pages = params.get(PAGES_PARAM).filter(|value| !value.is_null());
    let ranges = params
        .get(PAGE_RANGES_PARAM)
        .filter(|value| !value.is_null());
    if pages.is_some() && ranges.is_some() {
        return Err(Error::request_field(PAGES_PARAM, Refusal::PagesConflict));
    }
    let mapped_pages = pages_to_ranges(pages)?;
    let top_level_ranges = ranges.map(parse_page_ranges).transpose()?;
    if let Some(overrides) = params
        .get("extra_body")
        .and_then(|body| body.get(PAGE_RANGES_PARAM))
    {
        return parse_page_ranges(overrides);
    }
    Ok(top_level_ranges.or(mapped_pages).unwrap_or_default())
}

fn parse_page_ranges(value: &Value) -> Result<Vec<PageRange>, Error> {
    if value.is_null() {
        return Ok(Vec::new());
    }
    Vec::<PageRange>::deserialize(value)
        .ok()
        .filter(|ranges| {
            ranges
                .iter()
                .all(|range| 1 <= range.start && range.start <= range.end)
        })
        .ok_or_else(|| Error::request_field(PAGE_RANGES_PARAM, Refusal::PageRanges))
}

fn pages_to_ranges(value: Option<&Value>) -> Result<Option<Vec<PageRange>>, Error> {
    let invalid = || Error::request_field(PAGES_PARAM, Refusal::Pages);
    let pages = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(pages)) if pages.is_empty() => return Ok(None),
        Some(Value::Array(pages)) => pages,
        Some(_) => return Err(invalid()),
    };
    let pages = pages
        .iter()
        .map(|page| {
            page.as_i64()
                .filter(|index| *index >= 0)
                .and_then(|index| index.checked_add(1))
                .ok_or_else(invalid)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut ranges: Vec<PageRange> = Vec::new();
    for page in pages {
        match ranges.last_mut() {
            Some(last) if last.end.checked_add(1) == Some(page) => last.end = page,
            _ => ranges.push(PageRange {
                start: page,
                end: page,
            }),
        }
    }
    Ok(Some(ranges))
}

/// The page numbers Nace parses for `ranges`: in page order and once each, as Nace sorts and
/// merges the ranges, lazily, so a range that reaches far past the document allocates nothing.
fn requested_pages(ranges: &[PageRange]) -> impl Iterator<Item = i64> + use<> {
    requested_spans(ranges).flatten()
}

fn requested_spans(ranges: &[PageRange]) -> impl Iterator<Item = RangeInclusive<i64>> + use<> {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable_by_key(|range| range.start);
    sorted.into_iter().scan(1, |next: &mut i64, range| {
        let pages = range.start.max(*next)..=range.end;
        *next = (*next).max(range.end.saturating_add(1));
        Some(pages)
    })
}

fn requested_pages_in_document(ranges: &[PageRange], page_count: i64) -> i64 {
    requested_spans(ranges)
        .map(|span| {
            (*span.end())
                .min(page_count)
                .saturating_sub(*span.start())
                .saturating_add(1)
                .max(0)
        })
        .sum()
}

fn normalize_job(
    model: &str,
    job: ParseJob,
    full_markdown: Option<String>,
    page_ranges: &[PageRange],
) -> Result<LiteLLMOcrResponse, Error> {
    match (&job.status, job.error) {
        (JobStatus::Succeeded, _) => {}
        (
            JobStatus::Failed,
            Some(JobError {
                code,
                message,
                retryable: false,
            }),
        ) => {
            return Err(Error::RejectedDocument {
                provider: PROVIDER,
                message: clean_text(&format!("{code}: {message}"), &[]),
                guidance: "check that the file opens and that its type matches its contents",
            });
        }
        (status, Some(error)) => {
            return Err(Error::OperationStatus(clean_text(
                &format!("{status} ({}: {})", error.code, error.message),
                &[],
            )));
        }
        (status, None) => return Err(Error::OperationStatus(clean_text(&status.to_string(), &[]))),
    }
    let Some(result) = job.result else {
        return Err(match job.result_state.as_deref() {
            Some(NOT_RETAINED) => Error::RejectedDocument {
                provider: PROVIDER,
                message: "the account keeps no results (zero retention) and the parse ran past the wait of the create call, so Nace did not keep the result".into(),
                guidance: "set a request timeout of at least 120 seconds so the create call waits the full 60 seconds, or parse fewer pages per call",
            },
            _ => Error::ResponseField {
                path: "result".into(),
                cause: None,
            },
        });
    };
    let document = result.document;
    let markdown = full_markdown.or(document.markdown).unwrap_or_default();
    // With `page_ranges`, the page count can cover pages that were not requested, so only the
    // billed units count the pages, and a result without them stays one page.
    let processed = match page_ranges.is_empty() {
        true => job.units.filter(|units| *units > 0).or(document.page_count),
        false => job.units,
    };
    let expected: Vec<usize> = processed
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| *count > 0)
        .into_iter()
        .collect();
    let parsed_pages = match page_ranges.is_empty() {
        true => document.page_count,
        false => document
            .page_count
            .map(|count| requested_pages_in_document(page_ranges, count)),
    };
    Ok(LiteLLMOcrResponse {
        usage_info: Some(OcrUsageInfo {
            pages_processed: job.units.or(parsed_pages),
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(
            model,
            pages(&markdown, document.blocks, &expected, page_ranges),
        )
    })
}

/// A `Bearer` credential or a Nace key, masked whole.
static CREDENTIAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b(?:bearer\s+|nace_sk_)[^\s"'<>]*"#)
        .expect("the credential pattern compiles")
});

/// The text of a Nace API error: only the `error.code` and `error.message` of its envelope,
/// cleaned by `clean_text`, and a fixed text when the body has neither.
fn api_error_text(body: &str, secrets: &[&str]) -> String {
    let error = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|body| body.get("error").cloned());
    let field = |name: &str| -> Option<String> {
        error
            .as_ref()?
            .get(name)?
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .map(str::to_owned)
    };
    match (field("code"), field("message")) {
        (Some(code), Some(message)) => clean_text(&format!("{code}: {message}"), secrets),
        (Some(text), None) | (None, Some(text)) => clean_text(&text, secrets),
        (None, None) => "Nace sent no error code or message".to_owned(),
    }
}

/// `text` with every copy of each secret, every `Bearer` credential and every Nace key masked,
/// then the query and fragment cut from every URL, since a signed link carries its signature
/// there, then capped at `MAX_MESSAGE_CHARS` characters.
fn clean_text(text: &str, secrets: &[&str]) -> String {
    let text = secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .fold(text.to_owned(), |text, secret| text.replace(secret, "****"));
    let text = CREDENTIAL.replace_all(&text, "****");
    let mut rest = text.as_ref();
    let mut cleaned = String::with_capacity(rest.len());
    while let Some(scheme) = rest.find("://") {
        let url_end = scheme + url_length(&rest[scheme..]);
        let kept = rest[scheme..url_end]
            .find(['?', '#'])
            .map_or(url_end, |cut| scheme + cut);
        cleaned.push_str(&rest[..kept]);
        rest = &rest[url_end..];
    }
    cleaned.push_str(rest);
    match cleaned.char_indices().nth(MAX_MESSAGE_CHARS) {
        Some((cut, _)) => format!("{}...", &cleaned[..cut]),
        None => cleaned,
    }
}

/// Where a URL that starts `text` ends: at whitespace, a quote, `<`, `>`, a backtick or a
/// backslash.
fn url_length(text: &str) -> usize {
    text.find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '`' | '\\'))
        .unwrap_or(text.len())
}

/// One page per Nace page when the markdown splits cleanly (see `split_pages`), otherwise the
/// whole markdown as one page. A one-page result carries every block, a split one carries the
/// blocks located on each page.
fn pages(
    markdown: &str,
    blocks: Vec<Map<String, Value>>,
    expected: &[usize],
    page_ranges: &[PageRange],
) -> Vec<OcrPage> {
    let covered = CoveredText::new(&blocks);
    match split_pages(markdown, &covered, expected, page_ranges) {
        Some(split) if split.len() > 1 => {
            let mut by_page = blocks_by_page(blocks);
            split
                .into_iter()
                .map(|(index, text)| page(index, text, by_page.remove(&index).unwrap_or_default()))
                .collect()
        }
        Some(split) => split
            .into_iter()
            .map(|(index, text)| page(index, text, all_blocks(&blocks)))
            .collect(),
        None if markdown.trim().is_empty() && blocks.is_empty() => Vec::new(),
        None => vec![page(
            0,
            markdown.trim_matches(['\n', '\r']).to_owned(),
            all_blocks(&blocks),
        )],
    }
}

fn page(index: i64, markdown: String, blocks: Vec<Value>) -> OcrPage {
    OcrPage {
        index,
        markdown,
        extra_fields: (!blocks.is_empty())
            .then(|| ("blocks".to_owned(), Value::Array(blocks)))
            .into_iter()
            .collect(),
        ..Default::default()
    }
}

fn all_blocks(blocks: &[Map<String, Value>]) -> Vec<Value> {
    blocks.iter().cloned().map(Value::Object).collect()
}

fn blocks_by_page(blocks: Vec<Map<String, Value>>) -> BTreeMap<i64, Vec<Value>> {
    blocks
        .into_iter()
        .filter_map(|block| {
            let page = ["location", "markdown_span"]
                .iter()
                .find_map(|field| block.get(*field)?.get("page")?.as_i64())?;
            Some((page.checked_sub(1)?, Value::Object(block)))
        })
        .fold(BTreeMap::new(), |mut pages, (index, block)| {
            pages.entry(index).or_insert_with(Vec::new).push(block);
            pages
        })
}

/// The character ranges of the markdown that blocks cover (`markdown_span`), so a marker line
/// quoted inside a block's text is not taken for a page boundary.
struct CoveredText {
    starts: Vec<usize>,
    reach: Vec<usize>,
}

impl CoveredText {
    fn new(blocks: &[Map<String, Value>]) -> Self {
        let mut spans: Vec<(usize, usize)> = blocks
            .iter()
            .filter_map(|block| {
                let span = block.get("markdown_span")?;
                let offset = |field: &str| usize::try_from(span.get(field)?.as_u64()?).ok();
                let (start, end) = (offset("char_start")?, offset("char_end")?);
                (start < end).then_some((start, end))
            })
            .collect();
        spans.sort_unstable();
        let reach = spans
            .iter()
            .scan(0, |reach, (_, end)| {
                *reach = (*reach).max(*end);
                Some(*reach)
            })
            .collect();
        Self {
            starts: spans.into_iter().map(|(start, _)| start).collect(),
            reach,
        }
    }

    fn covers(&self, offset: usize) -> bool {
        let opened = self.starts.partition_point(|start| *start <= offset);
        opened > 0 && self.reach[opened - 1] > offset
    }
}

/// Splits Nace markdown into pages on its `--- Page N ---` lines in one pass, each page at index
/// N - 1. A marker line inside a block's text (`markdown_span`) is page content. Every other
/// marker line must continue one sequence that opens the markdown: the requested pages in order
/// when the request has `page_ranges`, since Nace keeps the document's page numbers, and
/// otherwise strictly consecutive numbers. The sequence must give as many pages as the job
/// processed (billed units, else its page count), and at most `MAX_PAGES`. Otherwise the result
/// is `None`, and the caller keeps the markdown as one page.
fn split_pages(
    markdown: &str,
    covered: &CoveredText,
    expected: &[usize],
    page_ranges: &[PageRange],
) -> Option<Vec<(i64, String)>> {
    let limit = expected.iter().copied().max()?.min(MAX_PAGES);
    let mut requested = (!page_ranges.is_empty()).then(|| requested_pages(page_ranges));
    let mut markers: Vec<(i64, usize, usize)> = Vec::new();
    let mut offset = 0;
    let mut chars = 0;
    for line in markdown.split_inclusive('\n') {
        if let Some(number) = page_marker(line)
            && !covered.covers(chars)
        {
            let next = match (requested.as_mut(), markers.last()) {
                (Some(pages), _) => pages.next(),
                (None, Some((last, ..))) => last.checked_add(1),
                (None, None) => Some(number),
            };
            if (markers.is_empty() && offset != 0) || next != Some(number) || markers.len() == limit
            {
                return None;
            }
            markers.push((number, offset, offset + line.len()));
        }
        offset += line.len();
        chars += line.chars().count();
    }
    if markers.is_empty() || !expected.contains(&markers.len()) {
        return None;
    }
    let ends = markers
        .iter()
        .skip(1)
        .map(|(_, start, _)| *start)
        .chain(std::iter::once(markdown.len()));
    Some(
        markers
            .iter()
            .zip(ends)
            .map(|((number, _, body), end)| {
                (
                    number - 1,
                    markdown[*body..end].trim_matches(['\n', '\r']).to_owned(),
                )
            })
            .collect(),
    )
}

fn page_marker(line: &str) -> Option<i64> {
    line.trim_end_matches(['\n', '\r'])
        .strip_prefix("--- Page ")?
        .strip_suffix(" ---")?
        .parse::<i64>()
        .ok()
        .filter(|number| *number >= 1)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn arguments(value: Value) -> CallArguments {
        serde_json::from_value(value).unwrap()
    }

    fn document(source: &str, document_name: Option<&str>) -> OcrDocument {
        OcrDocument::DocumentUrl {
            document_url: source.into(),
            extra_fields: document_name
                .map(|name| ("document_name".to_owned(), Some(name.to_owned())))
                .into_iter()
                .collect(),
        }
    }

    fn connection(api_base: Option<&str>, source: InputSource) -> OcrConnection {
        OcrConnection {
            api_key: Some(SecretValue::new("deployment-key")),
            api_key_source: InputSource::Deployment,
            api_base: api_base.map(str::to_owned),
            api_base_source: source,
            ..Default::default()
        }
    }

    fn blocks(value: Value) -> Vec<Map<String, Value>> {
        serde_json::from_value(value).unwrap()
    }

    fn split(markdown: &str, expected: &[usize]) -> Option<Vec<(i64, String)>> {
        split_pages(markdown, &CoveredText::new(&[]), expected, &[])
    }

    fn page_texts(pages: &[OcrPage]) -> Vec<(i64, &str)> {
        pages
            .iter()
            .map(|page| (page.index, page.markdown.as_str()))
            .collect()
    }

    #[rstest]
    #[case::low("parse-low", ParseMode::Low)]
    #[case::medium("parse-medium", ParseMode::Medium)]
    #[case::high("parse-high", ParseMode::High)]
    fn the_model_suffix_sets_the_parse_mode(#[case] model: &str, #[case] expected: ParseMode) {
        let params = NaceParseConfig
            .map_ocr_params(&CallArguments::default(), model)
            .unwrap();
        let body = serde_json::to_value(params.request(NaceSource::Url {
            url: "https://example.com/a.pdf".into(),
            file_name: "a.pdf".into(),
        }))
        .unwrap();
        assert_eq!(body["parse_mode"], expected.to_string());
    }

    #[rstest]
    #[case::unknown_effort("parse-max")]
    #[case::no_prefix("low")]
    #[case::other_operation("extract-low")]
    fn models_outside_the_three_efforts_are_refused(#[case] model: &str) {
        let error = NaceParseConfig
            .map_ocr_params(&CallArguments::default(), model)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidModel { .. }), "{error}");
        assert_eq!(error.http_status_code(), Some(400));
    }

    #[rstest]
    #[case::parse_mode(json!({"parse_mode": "high"}), "parse_mode")]
    #[case::parser(json!({"parser": "datalab"}), "parser")]
    #[case::source(json!({"source": {"type": "parse_result", "job_id": "job"}}), "source")]
    #[case::corpus(json!({"corpus_sources": []}), "corpus_sources")]
    #[case::extra_body_parse_mode(json!({"extra_body": {"parse_mode": "high"}}), "parse_mode")]
    #[case::extra_body_source(json!({"extra_body": {"source": {}}}), "source")]
    #[case::extra_body_output(json!({"extra_body": {"output": {}}}), "output")]
    #[case::extra_body_password(json!({"extra_body": {"password": "pw"}}), "extra_body.password")]
    fn params_that_decide_billing_or_the_source_are_refused(
        #[case] options: Value,
        #[case] field: &str,
    ) {
        let error = NaceParseConfig
            .map_ocr_params(&arguments(options), "parse-low")
            .unwrap_err();
        assert!(
            matches!(&error, Error::RequestField { path, .. } if path == field),
            "{error}"
        );
        assert_eq!(error.http_status_code(), Some(400));
    }

    #[rstest]
    #[case::text(json!({"page_ranges": "1-2"}))]
    #[case::float_end(json!({"page_ranges": [{"start": 1, "end": 2.5}]}))]
    #[case::missing_end(json!({"page_ranges": [{"start": 1}]}))]
    #[case::page_zero(json!({"page_ranges": [{"start": 0, "end": 1}]}))]
    #[case::reversed(json!({"page_ranges": [{"start": 3, "end": 2}]}))]
    #[case::one_bad_entry(json!({"page_ranges": [{"start": 1, "end": 1}, {"start": "4", "end": 5}]}))]
    #[case::extra_body(json!({"extra_body": {"page_ranges": [{"start": 1, "end": "x"}]}}))]
    fn page_ranges_that_do_not_decode_are_refused_before_the_upload(#[case] options: Value) {
        let error = NaceParseConfig
            .map_ocr_params(&arguments(options), "parse-low")
            .unwrap_err();
        assert!(
            matches!(&error, Error::RequestField { path, .. } if path == "page_ranges"),
            "{error}"
        );
        assert_eq!(error.http_status_code(), Some(400));
    }

    #[rstest]
    #[case::ranges(json!({"page_ranges": [{"start": 1, "end": 1}, {"start": 4, "end": 5}]}))]
    #[case::null(json!({"page_ranges": null}))]
    #[case::empty(json!({"page_ranges": []}))]
    fn valid_or_absent_page_ranges_are_accepted(#[case] options: Value) {
        assert!(
            NaceParseConfig
                .map_ocr_params(&arguments(options), "parse-low")
                .is_ok()
        );
    }

    #[rstest]
    #[case::single(json!({"pages": [0]}), Some(json!([{"start": 1, "end": 1}])))]
    #[case::sorted_merged_and_deduped(
        json!({"pages": [4, 0, 3, 0, 1]}),
        Some(json!([{"start": 1, "end": 2}, {"start": 4, "end": 5}]))
    )]
    #[case::largest_index(
        json!({"pages": [i64::MAX - 1]}),
        Some(json!([{"start": i64::MAX, "end": i64::MAX}]))
    )]
    #[case::empty(json!({"pages": []}), None)]
    #[case::null(json!({"pages": null}), None)]
    #[case::null_ranges(
        json!({"pages": [0], "page_ranges": null}),
        Some(json!([{"start": 1, "end": 1}]))
    )]
    #[case::extra_body_overrides(
        json!({"pages": [0], "extra_body": {"page_ranges": [{"start": 4, "end": 5}]}}),
        Some(json!([{"start": 4, "end": 5}]))
    )]
    #[case::extra_body_null(
        json!({"pages": [0], "extra_body": {"page_ranges": null}}),
        Some(Value::Null)
    )]
    fn pages_are_mapped_to_the_effective_page_ranges(
        #[case] options: Value,
        #[case] expected: Option<Value>,
    ) {
        let call = arguments(options);
        let params = NaceParseConfig.map_ocr_params(&call, "parse-low").unwrap();
        let request = params.request(NaceSource::Url {
            url: "https://example.com/report.pdf".into(),
            file_name: "report.pdf".into(),
        });
        let body = compose_body(
            &call,
            &request,
            NaceParseConfig.get_supported_ocr_params("parse-low"),
        )
        .unwrap();
        assert_eq!(body.get(PAGES_PARAM), None);
        assert_eq!(body.get(PAGE_RANGES_PARAM), expected.as_ref());
        assert_eq!(
            requested_page_ranges(&call).unwrap(),
            serde_json::from_value::<Option<Vec<PageRange>>>(expected.unwrap_or(Value::Null))
                .unwrap()
                .unwrap_or_default()
        );
    }

    #[rstest]
    #[case::negative(json!({"pages": [-1]}))]
    #[case::whole_float(json!({"pages": [1.0]}))]
    #[case::fractional(json!({"pages": [0.5]}))]
    #[case::string_element(json!({"pages": ["0"]}))]
    #[case::string(json!({"pages": "0,2"}))]
    #[case::boolean(json!({"pages": [true]}))]
    #[case::null_element(json!({"pages": [null]}))]
    #[case::overflow(json!({"pages": [i64::MAX]}))]
    #[case::unsigned_overflow(json!({"pages": [u64::MAX]}))]
    #[case::number(json!({"pages": 0}))]
    #[case::conflict(json!({"pages": [0], "page_ranges": [{"start": 1, "end": 1}]}))]
    #[case::empty_conflict(json!({"pages": [], "page_ranges": []}))]
    #[case::invalid_with_override(json!({"pages": [-1], "extra_body": {"page_ranges": null}}))]
    fn invalid_pages_are_refused_before_the_upload(#[case] options: Value) {
        let error = NaceParseConfig
            .map_ocr_params(&arguments(options), "parse-low")
            .unwrap_err();
        assert!(matches!(&error, Error::RequestField { path, .. } if path == "pages"));
        assert_eq!(error.http_status_code(), Some(400));
    }

    #[rstest]
    #[case::string(json!({"extra_body": "page_ranges=1"}))]
    #[case::list(json!({"extra_body": [1, 2]}))]
    fn an_extra_body_that_is_not_an_object_is_refused_before_the_upload(#[case] options: Value) {
        let error = NaceParseConfig
            .map_ocr_params(&arguments(options), "parse-low")
            .unwrap_err();
        assert_eq!(error.http_status_code(), Some(400), "{error}");
    }

    #[rstest]
    #[case::absent(json!({}), json!({"include_page_markers": true}))]
    #[case::markers_off(
        json!({"output": {"include_page_markers": false, "table_format": "markdown"}}),
        json!({"include_page_markers": true, "table_format": "markdown"})
    )]
    #[case::formats_without_markdown(
        json!({"output": {"formats": ["text"]}}),
        json!({"formats": ["text", "markdown"], "include_page_markers": true})
    )]
    #[case::formats_with_markdown(
        json!({"output": {"formats": ["markdown"]}}),
        json!({"formats": ["markdown"], "include_page_markers": true})
    )]
    #[case::image_links_dropped(
        json!({"output": {"include_images": true, "table_format": "html"}, "drop_params": true}),
        json!({"table_format": "html", "include_page_markers": true})
    )]
    #[case::image_links_off(
        json!({"output": {"include_images": false}}),
        json!({"include_images": false, "include_page_markers": true})
    )]
    fn output_always_keeps_markdown_and_page_markers(
        #[case] options: Value,
        #[case] expected: Value,
    ) {
        let params = NaceParseConfig
            .map_ocr_params(&arguments(options), "parse-low")
            .unwrap();
        assert_eq!(Value::Object(params.output), expected);
    }

    /// Other top-level options such as `image_limit` still go to
    /// Nace on a host that passes them, which Nace refuses as unknown fields.
    #[test]
    fn only_nace_options_are_mapped_into_the_parse_options() {
        let params = NaceParseConfig
            .map_ocr_params(
                &arguments(json!({
                    "page_ranges": [{"start": 1, "end": 2}],
                    "ocr": {"languages": ["en"]},
                    "image_limit": 2,
                    "include_image_base64": true,
                    "drop_params": true
                })),
                "parse-low",
            )
            .unwrap();
        assert_eq!(
            Value::Object(params.options),
            json!({"page_ranges": [{"start": 1, "end": 2}], "ocr": {"languages": ["en"]}})
        );
    }

    #[rstest]
    #[case::boolean_false(json!({"include_image_base64": false}))]
    #[case::dropped(json!({"include_image_base64": true, "drop_params": true}))]
    #[case::null(json!({"include_image_base64": null}))]
    fn include_image_base64_is_accepted_and_never_sent(#[case] options: Value) {
        let call = arguments(options);
        let params = NaceParseConfig.map_ocr_params(&call, "parse-low").unwrap();
        let request = params.request(NaceSource::Url {
            url: "https://example.com/report.pdf".into(),
            file_name: "report.pdf".into(),
        });
        let body = compose_body(
            &call,
            &request,
            NaceParseConfig.get_supported_ocr_params("parse-low"),
        )
        .unwrap();
        assert_eq!(body.get(INCLUDE_IMAGE_BASE64_PARAM), None);
        assert_eq!(body.get(DROP_PARAMS_PARAM), None);
    }

    #[rstest]
    #[case::base64_alone(json!({"include_image_base64": true}), "include_image_base64")]
    #[case::base64_drop_params_off(
        json!({"include_image_base64": true, "drop_params": false}),
        "include_image_base64"
    )]
    #[case::links_alone(json!({"output": {"include_images": true}}), "output.include_images")]
    #[case::links_drop_params_off(
        json!({"output": {"include_images": true}, "drop_params": false}),
        "output.include_images"
    )]
    fn image_requests_are_refused_unless_params_are_dropped(
        #[case] options: Value,
        #[case] refused: &str,
    ) {
        let error = NaceParseConfig
            .map_ocr_params(&arguments(options), "parse-low")
            .unwrap_err();
        assert!(
            matches!(&error, Error::UnsupportedParam { param, .. } if param == refused),
            "{error}"
        );
    }

    #[rstest]
    #[case::path_extension("https://example.com/files/report.pdf", None, Some("report.pdf"))]
    #[case::upper_case_extension("https://example.com/SCAN.PNG", None, Some("SCAN.PNG"))]
    #[case::document_name("https://example.com/download?id=7", Some("q3.xlsx"), Some("q3.xlsx"))]
    #[case::no_extension("https://example.com/download?id=7", None, None)]
    #[case::unreadable_extension("https://example.com/page.php", None, None)]
    #[case::plain_http("http://example.com/report.pdf", None, None)]
    #[case::data_uri("data:application/pdf;base64,YWJj", Some("a.pdf"), None)]
    fn only_https_links_with_a_readable_extension_become_url_sources(
        #[case] source: &str,
        #[case] document_name: Option<&str>,
        #[case] file_name: Option<&str>,
    ) {
        let expected = file_name.map(|file_name| NaceSource::Url {
            url: source.into(),
            file_name: file_name.into(),
        });
        assert_eq!(url_source(&document(source, document_name)), expected);
    }

    #[rstest]
    #[case::pdf("application/pdf", ".pdf")]
    #[case::docx(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx"
    )]
    fn upload_paths_are_unique_and_keep_a_known_extension(
        #[case] mime_type: &str,
        #[case] extension: &str,
    ) {
        let first = upload_path(mime_type);
        assert!(first.starts_with("litellm/"), "{first}");
        assert!(first.ends_with(extension), "{first}");
        assert_ne!(first, upload_path(mime_type));
    }

    #[test]
    fn an_unknown_mime_type_uploads_without_an_extension() {
        let path = upload_path("application/x-unknown");
        assert!(!path.rsplit('/').next().unwrap().contains('.'), "{path}");
    }

    #[rstest]
    #[case::default_timeout(Duration::from_secs(6000), MAX_WAIT_SECONDS)]
    #[case::short_timeout(Duration::from_secs(30), 15)]
    #[case::tiny_timeout(Duration::from_millis(900), MIN_WAIT_SECONDS)]
    fn the_server_wait_leaves_room_in_the_request_timeout(
        #[case] timeout: Duration,
        #[case] expected: u64,
    ) {
        assert_eq!(wait_seconds(timeout), expected);
    }

    #[rstest]
    #[case::three_pages(
        "--- Page 1 ---\nfirst\n\n--- Page 2 ---\nsecond\n\n--- Page 3 ---\nthird",
        &[3],
        Some(vec![(0, "first"), (1, "second"), (2, "third")])
    )]
    #[case::crlf_lines(
        "--- Page 1 ---\r\nfirst\r\n\r\n--- Page 2 ---\r\nsecond",
        &[2],
        Some(vec![(0, "first"), (1, "second")])
    )]
    #[case::page_range(
        "--- Page 4 ---\nfour\n\n--- Page 5 ---\nfive",
        &[2],
        Some(vec![(3, "four"), (4, "five")])
    )]
    #[case::empty_last_page(
        "--- Page 1 ---\n--- Page two ---\n\n--- Page 2 ---\n",
        &[2],
        Some(vec![(0, "--- Page two ---"), (1, "")])
    )]
    #[case::quoted_earlier_marker(
        "--- Page 1 ---\none\n--- Page 1 ---\nquoted\n\n--- Page 2 ---\ntwo",
        &[2],
        None
    )]
    #[case::no_markers("# Release notes\n\nbody\n", &[1], None)]
    #[case::empty("", &[1], None)]
    #[case::newlines_only("\n\n\n", &[1], None)]
    #[case::text_before_the_first_marker("preface\n--- Page 1 ---\none", &[1], None)]
    #[case::count_below_the_job(
        "--- Page 1 ---\none\n--- Page 2 ---\ntwo",
        &[3],
        None
    )]
    #[case::forward_quoted_marker(
        "--- Page 1 ---\nsee\n--- Page 2 ---\nbelow\n--- Page 2 ---\ntwo",
        &[2],
        None
    )]
    #[case::unknown_page_count("--- Page 1 ---\none", &[], None)]
    fn markdown_is_split_only_on_a_clean_marker_sequence(
        #[case] markdown: &str,
        #[case] expected_count: &[usize],
        #[case] expected: Option<Vec<(i64, &str)>>,
    ) {
        let pages = split(markdown, expected_count);
        let pages = pages.as_ref().map(|pages| {
            pages
                .iter()
                .map(|(index, markdown)| (*index, markdown.as_str()))
                .collect::<Vec<_>>()
        });
        assert_eq!(pages, expected);
    }

    const GAP: &str = "--- Page 1 ---\none\n\n--- Page 4 ---\nfour\n\n--- Page 5 ---\nfive";

    #[rstest]
    #[case::gap(
        GAP,
        json!([{"start": 1, "end": 1}, {"start": 4, "end": 5}]),
        &[3],
        Some(vec![(0, "one"), (3, "four"), (4, "five")])
    )]
    #[case::unordered_ranges(
        GAP,
        json!([{"start": 4, "end": 5}, {"start": 1, "end": 1}]),
        &[3],
        Some(vec![(0, "one"), (3, "four"), (4, "five")])
    )]
    #[case::overlapping_ranges(
        "--- Page 2 ---\ntwo\n--- Page 3 ---\nthree",
        json!([{"start": 2, "end": 3}, {"start": 3, "end": 3}]),
        &[2],
        Some(vec![(1, "two"), (2, "three")])
    )]
    #[case::range_past_the_last_page(
        "--- Page 15 ---\nlast",
        json!([{"start": 15, "end": 40}]),
        &[1],
        Some(vec![(14, "last")])
    )]
    #[case::unbounded_range(
        "--- Page 1 ---\none\n--- Page 2 ---\ntwo",
        json!([{"start": 1, "end": i64::MAX}]),
        &[2],
        Some(vec![(0, "one"), (1, "two")])
    )]
    #[case::page_outside_the_ranges(
        "--- Page 1 ---\none\n--- Page 2 ---\ntwo\n--- Page 5 ---\nfive",
        json!([{"start": 1, "end": 1}, {"start": 4, "end": 5}]),
        &[3],
        None
    )]
    #[case::requested_page_missing(
        "--- Page 1 ---\none\n--- Page 5 ---\nfive",
        json!([{"start": 1, "end": 1}, {"start": 4, "end": 5}]),
        &[2],
        None
    )]
    #[case::quoted_marker_in_the_text(
        "--- Page 1 ---\nsee\n--- Page 4 ---\nquoted\n--- Page 4 ---\nfour\n--- Page 5 ---\nfive",
        json!([{"start": 1, "end": 1}, {"start": 4, "end": 5}]),
        &[3],
        None
    )]
    #[case::gap_without_ranges(GAP, json!([]), &[3], None)]
    #[case::largest_marker_number(
        "--- Page 9223372036854775807 ---\none\n--- Page 9223372036854775807 ---\ntwo",
        json!([]),
        &[2],
        None
    )]
    fn requested_page_ranges_set_the_marker_sequence(
        #[case] markdown: &str,
        #[case] page_ranges: Value,
        #[case] expected_count: &[usize],
        #[case] expected: Option<Vec<(i64, &str)>>,
    ) {
        let page_ranges: Vec<PageRange> = serde_json::from_value(page_ranges).unwrap();
        let pages = split_pages(
            markdown,
            &CoveredText::new(&[]),
            expected_count,
            &page_ranges,
        );
        let pages = pages.as_ref().map(|pages| {
            pages
                .iter()
                .map(|(index, markdown)| (*index, markdown.as_str()))
                .collect::<Vec<_>>()
        });
        assert_eq!(pages, expected);
    }

    #[test]
    fn a_marker_quoted_inside_a_block_stays_in_its_page() {
        let markdown = "--- Page 1 ---\nsee\n--- Page 2 ---\nquote\n\n--- Page 2 ---\ntwo";
        let quoted = markdown.find("see").unwrap();
        let second = markdown.rfind("--- Page 2").unwrap();
        let covered = CoveredText::new(&blocks(json!([
            {"markdown_span": {"char_start": quoted, "char_end": second - 2, "page": 1}},
            {"markdown_span": {"char_start": second + 15, "char_end": markdown.len(), "page": 2}}
        ])));
        let pages = split_pages(markdown, &covered, &[2], &[]).unwrap();
        assert_eq!(
            pages,
            [
                (0, "see\n--- Page 2 ---\nquote".to_owned()),
                (1, "two".to_owned())
            ]
        );
    }

    #[test]
    fn block_spans_count_characters_not_bytes() {
        let markdown = "--- Page 1 ---\nÜbersicht\n--- Page 2 ---\nzwei";
        let marker = markdown[..markdown.rfind("--- Page").unwrap()]
            .chars()
            .count();
        // The span opens one character after the marker, which is inside it in bytes only.
        let covered = CoveredText::new(&blocks(json!([
            {"markdown_span": {"char_start": marker + 1, "char_end": marker + 20}}
        ])));
        assert_eq!(split_pages(markdown, &covered, &[2], &[]).unwrap().len(), 2);
    }

    #[test]
    fn the_split_stops_at_the_job_page_count() {
        let markdown = (1..=4)
            .map(|page| format!("--- Page {page} ---\ntext {page}\n"))
            .collect::<String>();
        assert_eq!(split(&markdown, &[3]), None);
        assert_eq!(split(&markdown, &[4]).unwrap().len(), 4);
    }

    #[test]
    fn the_split_stops_at_the_absolute_page_limit() {
        let markdown = (1..=MAX_PAGES + 1)
            .map(|page| format!("--- Page {page} ---\n"))
            .collect::<String>();
        assert_eq!(split(&markdown, &[MAX_PAGES + 1]), None);
        let within = (1..=MAX_PAGES)
            .map(|page| format!("--- Page {page} ---\n"))
            .collect::<String>();
        assert_eq!(split(&within, &[MAX_PAGES]).unwrap().len(), MAX_PAGES);
    }

    #[test]
    fn blocks_are_grouped_onto_their_page() {
        let pages = pages(
            "--- Page 1 ---\none\n\n--- Page 2 ---\ntwo",
            blocks(json!([
                {"id": "a", "location": {"kind": "page_region", "page": 2}},
                {"id": "b", "location": {"kind": "page_region", "page": 1}},
                {"id": "c", "markdown_span": {"kind": "text_span", "page": 2}},
                {"id": "d", "location": {"kind": "sheet_range", "sheet": "S"}}
            ])),
            &[2],
            &[],
        );
        let ids = |page: &OcrPage| -> Vec<Value> {
            page.extra_fields["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|block| block["id"].clone())
                .collect()
        };
        assert_eq!(ids(&pages[0]), [json!("b")]);
        assert_eq!(ids(&pages[1]), [json!("a"), json!("c")]);
    }

    #[test]
    fn a_document_without_pages_keeps_every_block_on_its_one_page() {
        let pages = pages(
            "| a | b |\n| 1 | 2 |\n",
            blocks(json!([
                {"id": "a", "location": {"kind": "sheet_range", "sheet": "S"}},
                {"id": "b", "location": {"kind": "text_span", "page": null}}
            ])),
            &[1],
            &[],
        );
        assert_eq!(page_texts(&pages), [(0, "| a | b |\n| 1 | 2 |")]);
        assert_eq!(pages[0].extra_fields["blocks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn empty_markdown_keeps_its_blocks_on_one_page() {
        let pages = pages(
            "",
            blocks(json!([{"id": "a", "type": "figure"}])),
            &[1],
            &[],
        );
        assert_eq!(page_texts(&pages), [(0, "")]);
        assert_eq!(pages[0].extra_fields["blocks"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn empty_markdown_without_blocks_has_no_pages() {
        assert!(pages(" \n", Vec::new(), &[1], &[]).is_empty());
    }

    #[test]
    fn a_text_file_that_quotes_marker_lines_stays_one_page() {
        let text = "notes\n--- Page 1 ---\n--- Page 2 ---\nend";
        let pages = pages(text, Vec::new(), &[1], &[]);
        assert_eq!(page_texts(&pages), [(0, text)]);
    }

    #[rstest]
    #[case::units(json!(3), json!(1), json!([]), Some(3))]
    #[case::page_count_fallback(Value::Null, json!(2), json!([]), Some(2))]
    #[case::units_with_page_ranges(json!(4), json!(10), json!([{"start": 1, "end": 2}]), Some(4))]
    #[case::requested_pages_fallback(Value::Null, json!(300), json!([{"start": 1, "end": 2}]), Some(2))]
    #[case::overlapping_ranges(
        Value::Null,
        json!(10),
        json!([{"start": 2, "end": 4}, {"start": 1, "end": 3}]),
        Some(4)
    )]
    #[case::ranges_past_the_document(
        Value::Null,
        json!(3),
        json!([{"start": 2, "end": 5}, {"start": 9, "end": 9}]),
        Some(2)
    )]
    #[case::no_page_count(Value::Null, Value::Null, json!([{"start": 1, "end": 2}]), None)]
    fn pages_processed_comes_from_billed_units(
        #[case] units: Value,
        #[case] page_count: Value,
        #[case] page_ranges: Value,
        #[case] expected: Option<i64>,
    ) {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "succeeded", "units": units,
            "result": {"document": {"markdown": "text", "page_count": page_count}}
        }))
        .unwrap();
        let page_ranges: Vec<PageRange> = serde_json::from_value(page_ranges).unwrap();
        let response = normalize_job("parse-low", job, None, &page_ranges).unwrap();
        assert_eq!(response.usage_info.unwrap().pages_processed, expected);
    }

    #[rstest]
    #[case::non_retryable(
        json!({"status": "failed", "error": {"code": "corrupt_file", "message": "not a PDF", "retryable": false}}),
        Some(400)
    )]
    #[case::retryable(
        json!({"status": "failed", "error": {"code": "internal_error", "message": "try again", "retryable": true}}),
        None
    )]
    #[case::null_error_fields(
        json!({"status": "failed", "error": {"code": null, "message": null, "retryable": null}}),
        Some(400)
    )]
    #[case::cancelled(json!({"status": "cancelled"}), None)]
    #[case::unknown_status(json!({"status": "paused"}), None)]
    fn jobs_that_did_not_succeed_are_errors(#[case] job: Value, #[case] status: Option<u16>) {
        let job: ParseJob = serde_json::from_value(
            json!({"job_id": "job"})
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(job.as_object().unwrap().clone())
                .collect::<Map<String, Value>>()
                .into(),
        )
        .unwrap();
        let error = normalize_job("parse-low", job, None, &[]).unwrap_err();
        assert_eq!(error.http_status_code(), status, "{error}");
    }

    #[test]
    fn a_failed_job_reports_the_nace_code_and_message() {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "failed",
            "error": {"code": "corrupt_file", "message": "The uploaded file is not a readable PDF.", "retryable": false}
        }))
        .unwrap();
        let error = normalize_job("parse-low", job, None, &[]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("corrupt_file: The uploaded file is not a readable PDF."),
            "{error}"
        );
    }

    #[test]
    fn a_result_that_zero_retention_did_not_keep_is_a_bad_request_with_guidance() {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "succeeded", "units": 3,
            "result": null, "result_state": "not_retained"
        }))
        .unwrap();
        let error = normalize_job("parse-low", job, None, &[]).unwrap_err();
        assert_eq!(error.http_status_code(), Some(400));
        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains("zero retention"), "{report}");
        assert!(report.contains("request timeout"), "{report}");
    }

    #[test]
    fn a_missing_result_without_a_state_stays_a_response_error() {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "succeeded", "result": null
        }))
        .unwrap();
        let error = normalize_job("parse-low", job, None, &[]).unwrap_err();
        assert!(matches!(error, Error::ResponseField { .. }), "{error}");
    }

    #[rstest]
    #[case::signed_link(
        "fetch_failed: could not read https://bucket.example/doc.pdf?X-Amz-Signature=abc&X-Amz-Credential=key now",
        "fetch_failed: could not read https://bucket.example/doc.pdf now"
    )]
    #[case::fragment("see https://docs.example/a#token=1.", "see https://docs.example/a")]
    #[case::plain("corrupt_file: not a PDF", "corrupt_file: not a PDF")]
    #[case::quoted(
        r#"read "https://b.example/a.pdf?sig=1" failed"#,
        r#"read "https://b.example/a.pdf" failed"#
    )]
    fn clean_text_drops_url_queries(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(clean_text(text, &[]), expected);
    }

    #[rstest]
    #[case::bearer(
        "invalid header Authorization: Bearer abc.DEF-123 rejected",
        "invalid header Authorization: **** rejected"
    )]
    #[case::bearer_field_value(
        r#"token "Bearer SAMPLE_VALUE" rejected"#,
        r#"token "****" rejected"#
    )]
    #[case::nace_key("key nace_sk_live_9f8e7d is revoked", "key **** is revoked")]
    #[case::exact_secret("token tok-123-abc expired", "token **** expired")]
    #[case::secret_in_a_url_query(
        "see https://h.example/a?t=tok-123-abc",
        "see https://h.example/a"
    )]
    #[case::secret_past_the_cap(
        &format!("{} tok-123-abc", "x".repeat(MAX_MESSAGE_CHARS - 5)),
        &format!("{} ****", "x".repeat(MAX_MESSAGE_CHARS - 5))
    )]
    fn clean_text_masks_credentials(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(clean_text(text, &["tok-123-abc"]), expected);
    }

    #[rstest]
    #[case::signed_link_in_both(
        "fetch_failed https://b.example/a.pdf?sig=1",
        "could not read https://b.example/a.pdf?sig=1",
        "fetch_failed https://b.example/a.pdf",
        "could not read https://b.example/a.pdf"
    )]
    #[case::exact_key_in_both(
        "bad key-123",
        "key key-123 is revoked",
        "bad ****",
        "key **** is revoked"
    )]
    #[case::credentials_in_both(
        "Bearer abc.DEF",
        "key nace_sk_live_9f8e7d is revoked",
        "****",
        "key **** is revoked"
    )]
    fn without_source_drops_the_source_and_cleans_the_error_in_place(
        #[case] code: &str,
        #[case] message: &str,
        #[case] clean_code: &str,
        #[case] clean_message: &str,
    ) {
        let headers = [("Authorization".to_owned(), "Bearer key-123".to_owned())];
        let job = json!({
            "job_id": "job-1",
            "source": {"type": "url", "url": "https://b.example/a.pdf?sig=1"},
            "status": "failed",
            "error": {"code": code, "message": message, "retryable": false},
            "timing": {"elapsed_ms": 1}
        });

        let seen = without_source(&serde_json::to_vec(&job).unwrap(), &headers);

        let expected = json!({
            "job_id": "job-1",
            "status": "failed",
            "error": {"code": clean_code, "message": clean_message, "retryable": false},
            "timing": {"elapsed_ms": 1}
        });
        assert_eq!(
            String::from_utf8(seen).unwrap(),
            serde_json::to_string(&expected).unwrap()
        );
    }

    #[rstest]
    #[case::null_error(json!({"job_id": "job-1", "error": null}))]
    #[case::null_fields(json!({"job_id": "job-1", "error": {"code": null, "message": null}}))]
    #[case::no_error(json!({"job_id": "job-1", "status": "succeeded"}))]
    fn without_source_keeps_an_error_without_text(#[case] job: Value) {
        let headers = [("Authorization".to_owned(), "Bearer key-123".to_owned())];

        let seen = without_source(&serde_json::to_vec(&job).unwrap(), &headers);

        assert_eq!(serde_json::from_slice::<Value>(&seen).unwrap(), job);
    }

    #[test]
    fn clean_text_is_capped() {
        let text = "x".repeat(MAX_MESSAGE_CHARS * 10);
        let capped = clean_text(&text, &[]);
        assert_eq!(capped.chars().count(), MAX_MESSAGE_CHARS + 3);
        assert!(capped.ends_with("..."));
    }

    #[rstest]
    #[case::code_and_message(
        r#"{"error": {"code": "invalid_api_key", "message": "Invalid API key.", "detail": "secret"}}"#,
        "invalid_api_key: Invalid API key."
    )]
    #[case::code_only(r#"{"error": {"code": "unavailable"}}"#, "unavailable")]
    #[case::other_fields_dropped(
        r#"{"detail": "echo nace_sk_1 and tok"}"#,
        "Nace sent no error code or message"
    )]
    #[case::not_json("<html>bad gateway</html>", "Nace sent no error code or message")]
    fn api_error_text_keeps_only_the_code_and_message(#[case] body: &str, #[case] expected: &str) {
        assert_eq!(api_error_text(body, &[]), expected);
    }

    #[test]
    fn a_failed_job_message_is_sanitized() {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "failed",
            "error": {
                "code": "fetch_failed",
                "message": format!("https://bucket.example/a.pdf?sig=secret {}", "y".repeat(1000)),
                "retryable": false
            }
        }))
        .unwrap();
        let report =
            litellm_error::Report::new(&normalize_job("parse-low", job, None, &[]).unwrap_err())
                .to_string();
        assert!(!report.contains("sig=secret"), "{report}");
        assert!(!report.contains(&"y".repeat(MAX_MESSAGE_CHARS)), "{report}");
    }

    #[rstest]
    #[case::one_unit(json!(1))]
    #[case::no_units(Value::Null)]
    #[case::zero_units(json!(0))]
    fn a_quoted_marker_for_a_page_past_the_document_stays_text(#[case] units: Value) {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "succeeded", "units": units,
            "result": {
                "document": {
                    "markdown": "--- Page 1 ---\nsee\n--- Page 5 ---\nquoted",
                    "page_count": 2,
                    "blocks": null
                }
            }
        }))
        .unwrap();
        let ranges = [
            PageRange { start: 1, end: 1 },
            PageRange { start: 5, end: 5 },
        ];
        let response = normalize_job("parse-low", job, None, &ranges).unwrap();
        assert_eq!(
            page_texts(&response.pages),
            [(0, "--- Page 1 ---\nsee\n--- Page 5 ---\nquoted")]
        );
    }

    #[test]
    fn null_collections_in_a_result_read_as_empty() {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "succeeded", "units": 2,
            "result": {
                "document": {"markdown": "--- Page 1 ---\none\n--- Page 2 ---\ntwo", "page_count": 2, "blocks": null},
                "disclosures": null
            }
        }))
        .unwrap();
        assert_eq!(truncated_content(&job).unwrap(), None);
        let response = normalize_job("parse-low", job, None, &[]).unwrap();
        assert_eq!(page_texts(&response.pages), [(0, "one"), (1, "two")]);
        assert_eq!(response.usage_info.unwrap().pages_processed, Some(2));
    }

    #[rstest]
    #[case::sheets(json!([{"code": "content_truncated", "target": "spreadsheet_region"}]))]
    #[case::document_without_link(json!([{"code": "content_truncated", "target": "document_content"}]))]
    fn a_truncated_result_without_a_full_document_link_is_an_error(#[case] disclosures: Value) {
        let job = serde_json::from_value(json!({
            "job_id": "job", "status": "succeeded",
            "result": {"document": {"markdown": "preview"}, "disclosures": disclosures}
        }))
        .unwrap();
        assert!(truncated_content(&job).is_err());
    }

    #[rstest]
    #[case::request_base_with_deployment_key(
        Some("https://request.example"),
        InputSource::Request,
        false
    )]
    #[case::deployment_base(Some("https://deployment.example"), InputSource::Deployment, true)]
    #[case::default_base(None, InputSource::Deployment, true)]
    fn a_request_controlled_base_never_receives_an_operator_key(
        #[case] api_base: Option<&str>,
        #[case] source: InputSource,
        #[case] allowed: bool,
    ) {
        assert_eq!(
            resolve_headers(&connection(api_base, source)).is_ok(),
            allowed
        );
    }

    #[test]
    fn a_request_controlled_base_accepts_a_request_key() {
        let connection = OcrConnection {
            api_key: Some(SecretValue::new("request-key")),
            api_key_source: InputSource::Request,
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };
        assert_eq!(
            resolve_headers(&connection).unwrap()[0],
            ("Authorization".to_owned(), "Bearer request-key".to_owned())
        );
    }

    #[rstest]
    #[case::https_other_host("https://upload.example/v1/files", true)]
    #[case::same_origin_http("http://127.0.0.1:9/files", true)]
    #[case::http_other_host("http://upload.example/v1/files", false)]
    #[case::http_other_port("http://127.0.0.1:10/files", false)]
    #[case::credentials("https://user:pass@upload.example/v1/files", false)]
    #[case::same_origin_credentials("http://user:pass@127.0.0.1:9/files", false)]
    #[case::other_scheme("ftp://upload.example/v1/files", false)]
    fn uploads_go_only_to_https_or_the_api_base_origin(#[case] url: &str, #[case] allowed: bool) {
        let connection = connection(Some("http://127.0.0.1:9"), InputSource::Deployment);
        assert_eq!(upload_target(url, &connection).is_ok(), allowed);
    }

    #[rstest]
    #[case::job_file(
        "https://console.example/v1/documents/jobs/job-1/document-content/full",
        true
    )]
    #[case::other_job(
        "https://console.example/v1/documents/jobs/job-2/document-content/full",
        false
    )]
    #[case::job_prefix(
        "https://console.example/v1/documents/jobs/job-10/document-content/full",
        false
    )]
    #[case::other_path("https://console.example/v1/files/document-content", false)]
    #[case::dot_segments("https://console.example/v1/documents/jobs/job-1/../../files/a", false)]
    #[case::other_host(
        "https://cdn.example/v1/documents/jobs/job-1/document-content/full",
        false
    )]
    #[case::plain_http(
        "http://console.example/v1/documents/jobs/job-1/document-content/full",
        false
    )]
    #[case::credentials(
        "https://user:pass@console.example/v1/documents/jobs/job-1/document-content/full",
        false
    )]
    #[case::job_path_only("https://console.example/v1/documents/jobs/job-1", false)]
    #[case::job_path_slash("https://console.example/v1/documents/jobs/job-1/", false)]
    #[case::encoded_slash(
        "https://console.example/v1/documents/jobs/job-1/..%2fjob-2/content",
        false
    )]
    #[case::encoded_slash_upper("https://console.example/v1/documents/jobs/job-1/a%2Fb", false)]
    #[case::encoded_backslash("https://console.example/v1/documents/jobs/job-1/a%5Cb", false)]
    #[case::encoded_percent("https://console.example/v1/documents/jobs/job-1/a%252fb", false)]
    fn only_this_jobs_files_receive_the_api_key(#[case] url: &str, #[case] allowed: bool) {
        let connection = connection(Some("https://console.example"), InputSource::Deployment);
        assert_eq!(job_file_url(&connection, "job-1", url).is_ok(), allowed);
    }

    #[rstest]
    #[case::empty("")]
    #[case::blank(" ")]
    fn a_job_without_an_id_owns_no_files(#[case] job_id: &str) {
        let connection = connection(Some("https://console.example"), InputSource::Deployment);
        let url = "https://console.example/v1/documents/jobs/job-2/document-content/full";
        assert!(job_file_url(&connection, job_id, url).is_err());
    }

    #[rstest]
    #[case::job_file("https://console.example/v1/documents/jobs/job-1/content", true)]
    #[case::job_prefix("https://console.example/v1/documents/jobs/job-10/content", false)]
    #[case::base_query_kept_off_the_path(
        "https://console.example/v1/documents/jobs/job-1?tenant=x",
        false
    )]
    fn a_base_with_a_query_still_bounds_the_job_path(#[case] url: &str, #[case] allowed: bool) {
        let connection = connection(
            Some("https://console.example?tenant=x"),
            InputSource::Deployment,
        );
        assert_eq!(job_file_url(&connection, "job-1", url).is_ok(), allowed);
    }

    #[rstest]
    #[case::root(
        "https://console.example/proxy",
        "https://console.example/proxy/v1/documents/jobs/job-1/content",
        true
    )]
    #[case::outside_the_base_path(
        "https://console.example/proxy",
        "https://console.example/v1/documents/jobs/job-1/content",
        false
    )]
    fn a_base_path_is_part_of_the_job_path(
        #[case] base: &str,
        #[case] url: &str,
        #[case] allowed: bool,
    ) {
        let connection = connection(Some(base), InputSource::Deployment);
        assert_eq!(job_file_url(&connection, "job-1", url).is_ok(), allowed);
    }

    #[test]
    fn only_a_workspace_file_source_is_released() {
        let request = |source: Value| {
            litellm_http::outbound::json(
                "https://console.example/v1/documents/parse",
                &[],
                &json!({"source": source, "parse_mode": "low"}),
                None,
            )
            .unwrap()
        };
        let file = json!({"type": "workspace_file", "workspace_id": "w", "file_id": "f"});
        assert_eq!(
            NaceParseConfig.request_cleanup(&request(file.clone())),
            Some(file)
        );
        assert_eq!(
            NaceParseConfig.request_cleanup(&request(
                json!({"type": "url", "url": "https://a.example/a.pdf"})
            )),
            None
        );
    }

    #[rstest]
    #[case::root(
        "https://console.example",
        "https://console.example/v1/documents/parse"
    )]
    #[case::version_suffix(
        "https://console.example/v1",
        "https://console.example/v1/documents/parse"
    )]
    fn the_parse_url_accepts_a_base_with_or_without_v1(#[case] base: &str, #[case] expected: &str) {
        let connection = connection(Some(base), InputSource::Deployment);
        assert_eq!(
            api_url(&connection, &["v1", "documents", "parse"])
                .unwrap()
                .into_string(),
            expected
        );
    }

    mod create_response {
        use std::sync::{Arc, Mutex};

        use futures_util::future::BoxFuture;
        use litellm_host::interceptors::WireRequest;
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::TcpListener,
        };

        use super::*;

        const BAD_UNITS: &str = r#"{"job_id":"job-1","status":"running","units":"invalid"}"#;
        const RUNNING: &str = r#"{"job_id":"job-1","status":"running"}"#;
        const FAILED_WITH_KEY: &str = r#"{"job_id":"job-1","status":"failed","error":{"code":"bad_key key-123","message":"Nace saw Bearer key-123 and key-123","retryable":false}}"#;

        struct Hooks {
            reject: bool,
        }

        impl CallHooks<Error> for Hooks {
            fn before_provider_request(
                &self,
                wire: WireRequest,
            ) -> BoxFuture<'_, Result<WireRequest, Error>> {
                Box::pin(async move { Ok(wire) })
            }

            fn response_received<'a>(&'a self, _: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
                let reject = self.reject;
                Box::pin(async move {
                    match reject {
                        true => Err(Error::OperationStatus(
                            "rejected by the response hook".into(),
                        )),
                        false => Ok(()),
                    }
                })
            }
        }

        /// Hooks that keep every body the response hook receives.
        #[derive(Default)]
        struct Recorder {
            bodies: Mutex<Vec<String>>,
        }

        impl CallHooks<Error> for Recorder {
            fn before_provider_request(
                &self,
                wire: WireRequest,
            ) -> BoxFuture<'_, Result<WireRequest, Error>> {
                Box::pin(async move { Ok(wire) })
            }

            fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
                self.bodies
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(body).into_owned());
                Box::pin(async move { Ok(()) })
            }
        }

        /// A local Nace that answers the n-th request with the n-th of `bodies` and every later
        /// one with a 204, and records each request line.
        async fn local_nace(bodies: &'static [&'static str]) -> (String, Arc<Mutex<Vec<String>>>) {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let seen = Arc::new(Mutex::new(Vec::new()));
            let recorder = seen.clone();
            tokio::spawn(async move {
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut request = vec![0; 8192];
                    let read = socket.read(&mut request).await.unwrap();
                    let line = String::from_utf8_lossy(&request[..read])
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_owned();
                    let answer = {
                        let mut seen = recorder.lock().unwrap();
                        seen.push(line);
                        bodies.get(seen.len() - 1)
                    };
                    let response = match answer {
                        Some(body) => format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        ),
                        None => "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n".to_owned(),
                    };
                    socket.write_all(response.as_bytes()).await.unwrap();
                }
            });
            (base, seen)
        }

        /// A local Nace that answers every request with a 204 after `delay`.
        async fn slow_nace(delay: Duration) -> String {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            tokio::spawn(async move {
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    tokio::spawn(async move {
                        let mut request = vec![0; 8192];
                        if socket.read(&mut request).await.unwrap_or(0) == 0 {
                            return;
                        }
                        tokio::time::sleep(delay).await;
                        let response = "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n";
                        let _ = socket.write_all(response.as_bytes()).await;
                    });
                }
            });
            base
        }

        #[tokio::test]
        async fn a_cleanup_call_outlasts_a_short_caller_timeout() {
            let base = slow_nace(Duration::from_millis(300)).await;
            let connection = OcrConnection {
                timeout: Duration::from_millis(50),
                ..connection(Some(&base), InputSource::Deployment)
            };
            let headers = [("Authorization".to_owned(), "Bearer key".to_owned())];

            let deleted = delete(
                &OcrClient::for_test(),
                &connection,
                &headers,
                &["v1", "documents", "jobs", "job-1"],
            )
            .await;

            assert!(deleted.is_ok(), "{:?}", deleted.err());
        }

        #[rstest]
        #[case::decode_error(false, "units")]
        #[case::hook_rejection_wins(true, "rejected by the response hook")]
        #[tokio::test]
        async fn a_create_response_that_fails_still_deletes_its_job(
            #[case] reject: bool,
            #[case] reported: &str,
        ) {
            let (base, seen) = local_nace(&[BAD_UNITS]).await;
            let client = OcrClient::for_test();
            let created = client
                .polling_http()
                .post(format!("{base}/v1/documents/parse"))
                .send()
                .await
                .unwrap();
            let connection = connection(Some(&base), InputSource::Deployment);
            let headers = [("Authorization".to_owned(), "Bearer key".to_owned())];

            let error = NaceParseConfig
                .async_transform_ocr_response(
                    "parse-low",
                    created,
                    OcrResponseContext {
                        client: &client,
                        connection: &connection,
                        hooks: &Hooks { reject },
                        request_format: OcrResponseFormat::Litellm,
                        url: "",
                        headers: &headers,
                        optional_params: &CallArguments::default(),
                    },
                )
                .await
                .unwrap_err();

            let report = litellm_error::Report::new(&error).to_string();
            assert!(report.contains(reported), "{report}");
            assert_eq!(
                *seen.lock().unwrap(),
                [
                    "POST /v1/documents/parse HTTP/1.1",
                    "DELETE /v1/documents/jobs/job-1 HTTP/1.1"
                ]
            );
        }

        #[rstest]
        #[case::create(&[FAILED_WITH_KEY], 1)]
        #[case::poll(&[RUNNING, FAILED_WITH_KEY], 2)]
        #[tokio::test]
        async fn the_response_hook_gets_job_errors_with_the_api_key_masked(
            #[case] answers: &'static [&'static str],
            #[case] hook_calls: usize,
        ) {
            let (base, _) = local_nace(answers).await;
            let client = OcrClient::for_test();
            let created = client
                .polling_http()
                .post(format!("{base}/v1/documents/parse"))
                .send()
                .await
                .unwrap();
            let connection = connection(Some(&base), InputSource::Deployment);
            let headers = [("Authorization".to_owned(), "Bearer key-123".to_owned())];
            let hooks = Recorder::default();

            let result = NaceParseConfig
                .async_transform_ocr_response(
                    "parse-low",
                    created,
                    OcrResponseContext {
                        client: &client,
                        connection: &connection,
                        hooks: &hooks,
                        request_format: OcrResponseFormat::Litellm,
                        url: "",
                        headers: &headers,
                        optional_params: &CallArguments::default(),
                    },
                )
                .await;

            assert!(result.is_err());
            let bodies = hooks.bodies.lock().unwrap();
            assert_eq!(bodies.len(), hook_calls, "{bodies:?}");
            let failed: Value = serde_json::from_str(bodies.last().unwrap()).unwrap();
            assert_eq!(
                failed["error"],
                json!({"code": "bad_key ****", "message": "Nace saw **** and ****", "retryable": false})
            );
        }
    }
}
