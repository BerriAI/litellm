use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::CoreResult;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::Url;
use serde_json::{Map, Value};
use url::Host;

use litellm_core::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use litellm_core::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use litellm_core::providers::reducto::ocr::transformation::{
    REDUCTO_PARSE_LEGACY_CONFIG, REDUCTO_PARSE_V3_CONFIG,
};
use litellm_core::providers::vertex_ai::ocr::transformation as vertex_ai;
use litellm_core::providers::vertex_ai::ocr::transformation::{
    VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG,
};

use super::http_client;

const ERROR_BODY_MAX_CHARS: usize = 256;
const AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB: f64 = 50.0;
const MAX_SAFE_FETCH_REDIRECTS: usize = 10;
const ERROR_BODY_MAX_BYTES: usize = ERROR_BODY_MAX_CHARS * 4;
const MAX_DOCUMENT_CLIENTS: usize = 128;
const DOCUMENT_FETCH_TIMEOUT_SECS: u64 = 600;

pub(super) fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub(super) async fn read_error_body(mut response: reqwest::Response) -> String {
    let mut collected: Vec<u8> = Vec::new();
    while collected.len() < ERROR_BODY_MAX_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let take = (ERROR_BODY_MAX_BYTES - collected.len()).min(chunk.len());
                collected.extend_from_slice(&chunk[..take]);
                if take < chunk.len() {
                    break;
                }
            }
            Ok(None) | Err(_) => break,
        }
    }
    truncate_error_body(&String::from_utf8_lossy(&collected))
}

pub(super) fn ocr_provider_config(
    provider: &str,
    model: &str,
) -> Option<&'static dyn OcrProviderConfig> {
    match provider {
        "mistral" => Some(&MISTRAL_OCR_CONFIG),
        "azure_ai" if is_azure_document_intelligence_model(model) => {
            Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG)
        }
        "azure_ai/doc-intelligence" => Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG),
        "azure_ai" => Some(&AZURE_AI_OCR_CONFIG),
        "vertex_ai" if vertex_ai::is_deepseek_model(model) => Some(&VERTEX_AI_DEEPSEEK_OCR_CONFIG),
        "vertex_ai" => Some(&VERTEX_AI_OCR_CONFIG),
        "reducto" if model == "parse-v3" => Some(&REDUCTO_PARSE_V3_CONFIG),
        "reducto" if model == "parse-legacy" => Some(&REDUCTO_PARSE_LEGACY_CONFIG),
        _ => None,
    }
}

fn is_azure_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "OCR extra_headers.{key} must be a string, got {}",
                        litellm_core::error::json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub(super) fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

fn document_url_field(document: &Value) -> CoreResult<Option<(&str, &str)>> {
    let Some(object) = document.as_object() else {
        return Ok(None);
    };
    let Some(doc_type) = object.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let field = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        _ => return Ok(None),
    };
    let Some(url) = object.get(field).and_then(Value::as_str) else {
        return Ok(None);
    };
    Ok(Some((field, url)))
}

fn is_url_requiring_fetch(url: &str) -> bool {
    !url.starts_with("data:") && (url.starts_with("http://") || url.starts_with("https://"))
}

fn max_document_download_bytes() -> u64 {
    let max_size_mb = std::env::var("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB);
    (max_size_mb.max(0.0) * 1024.0 * 1024.0) as u64
}

fn ipv4_in_cidr(ip: Ipv4Addr, network: Ipv4Addr, prefix_length: u32) -> bool {
    let mask = u32::MAX << (32 - prefix_length);
    u32::from(ip) & mask == u32::from(network) & mask
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_multicast()
                || ip.is_unspecified()
                || ipv4_in_cidr(ip, Ipv4Addr::new(0, 0, 0, 0), 8)
                || ipv4_in_cidr(ip, Ipv4Addr::new(100, 64, 0, 0), 10)
                || ipv4_in_cidr(ip, Ipv4Addr::new(192, 0, 0, 0), 24)
                || ipv4_in_cidr(ip, Ipv4Addr::new(192, 0, 2, 0), 24)
                || ipv4_in_cidr(ip, Ipv4Addr::new(192, 88, 99, 0), 24)
                || ipv4_in_cidr(ip, Ipv4Addr::new(198, 18, 0, 0), 15)
                || ipv4_in_cidr(ip, Ipv4Addr::new(198, 51, 100, 0), 24)
                || ipv4_in_cidr(ip, Ipv4Addr::new(203, 0, 113, 0), 24)
                || ipv4_in_cidr(ip, Ipv4Addr::new(240, 0, 0, 0), 4)
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            let first_segment = segments[0];
            let is_unique_local = (first_segment & 0xfe00) == 0xfc00;
            let is_link_local = (first_segment & 0xffc0) == 0xfe80;
            let is_site_local = (first_segment & 0xffc0) == 0xfec0;
            let is_documentation = first_segment == 0x2001 && segments[1] == 0x0db8;
            let is_discard =
                first_segment == 0x0100 && segments[1] == 0 && segments[2] == 0 && segments[3] == 0;
            let is_teredo = first_segment == 0x2001 && segments[1] == 0x0000;
            let is_benchmarking =
                first_segment == 0x2001 && segments[1] == 0x0002 && segments[2] == 0x0000;
            let is_orchid = first_segment == 0x2001 && (segments[1] & 0xfff0) == 0x0010;
            let is_orchidv2 = first_segment == 0x2001 && (segments[1] & 0xfff0) == 0x0020;
            let is_nat64_wellknown = first_segment == 0x0064
                && segments[1] == 0xff9b
                && segments[2] == 0
                && segments[3] == 0
                && segments[4] == 0
                && segments[5] == 0;
            let is_nat64_local =
                first_segment == 0x0064 && segments[1] == 0xff9b && segments[2] == 1;
            let is_6to4_embedded_blocked =
                first_segment == 0x2002 && embedded_ipv4_blocked(segments[1], segments[2]);
            let is_nat64_embedded_blocked =
                is_nat64_wellknown && embedded_ipv4_blocked(segments[6], segments[7]);
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || is_unique_local
                || is_link_local
                || is_site_local
                || is_documentation
                || is_discard
                || is_teredo
                || is_benchmarking
                || is_orchid
                || is_orchidv2
                || is_nat64_local
                || is_nat64_embedded_blocked
                || is_6to4_embedded_blocked
                || ip
                    .to_ipv4_mapped()
                    .or_else(|| ip.to_ipv4())
                    .map(|v4| is_blocked_ip(IpAddr::V4(v4)))
                    .unwrap_or(false)
        }
    }
}

fn embedded_ipv4_blocked(hi: u16, lo: u16) -> bool {
    let v4 = Ipv4Addr::new(
        (hi >> 8) as u8,
        (hi & 0xff) as u8,
        (lo >> 8) as u8,
        (lo & 0xff) as u8,
    );
    is_blocked_ip(IpAddr::V4(v4))
}

fn blocked_url_error() -> CoreError {
    CoreError::InvalidRequest("OCR document URL rejected by SSRF protection".to_string())
}

async fn pin_validated_url(url: &Url) -> CoreResult<Vec<SocketAddr>> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(blocked_url_error());
    }
    let port = url.port_or_known_default().ok_or_else(blocked_url_error)?;
    let addresses: Vec<SocketAddr> = match url.host() {
        Some(Host::Ipv4(ip)) => vec![SocketAddr::from((ip, port))],
        Some(Host::Ipv6(ip)) => vec![SocketAddr::from((ip, port))],
        Some(Host::Domain(domain)) => tokio::net::lookup_host((domain, port))
            .await
            .map_err(|_| blocked_url_error())?
            .collect(),
        None => return Err(blocked_url_error()),
    };
    if addresses.is_empty() || addresses.iter().any(|address| is_blocked_ip(address.ip())) {
        return Err(blocked_url_error());
    }
    Ok(addresses)
}

type PinnedAddrs = Arc<Mutex<HashMap<String, Vec<SocketAddr>>>>;

#[derive(Debug, Clone)]
struct PinnedResolver {
    pins: PinnedAddrs,
}

impl Resolve for PinnedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let pins = self.pins.clone();
        Box::pin(async move {
            let pinned = pins
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(name.as_str())
                .cloned();
            match pinned {
                Some(addresses) if !addresses.is_empty() => {
                    Ok(Box::new(addresses.into_iter()) as Addrs)
                }
                _ => Err(Box::<dyn std::error::Error + Send + Sync>::from(
                    "OCR document host was not pinned to a validated address",
                )),
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DocumentClientKey {
    scheme: String,
    host: String,
    port: u16,
    addresses: Vec<SocketAddr>,
}

fn document_client_cache() -> &'static Mutex<HashMap<DocumentClientKey, reqwest::Client>> {
    static CACHE: OnceLock<Mutex<HashMap<DocumentClientKey, reqwest::Client>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn document_fetch_client(url: &Url, addresses: &[SocketAddr]) -> CoreResult<reqwest::Client> {
    let host = url.host_str().ok_or_else(blocked_url_error)?.to_owned();
    let port = url.port_or_known_default().ok_or_else(blocked_url_error)?;
    let mut sorted = addresses.to_vec();
    sorted.sort();
    let key = DocumentClientKey {
        scheme: url.scheme().to_owned(),
        host: host.clone(),
        port,
        addresses: sorted.clone(),
    };
    let cache = document_client_cache();
    if let Some(client) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
        .cloned()
    {
        return Ok(client);
    }

    let mut pins = HashMap::new();
    pins.insert(host, sorted);
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(DOCUMENT_FETCH_TIMEOUT_SECS))
        .dns_resolver(Arc::new(PinnedResolver {
            pins: Arc::new(Mutex::new(pins)),
        }))
        .build()
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let mut guard = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.len() >= MAX_DOCUMENT_CLIENTS {
        if let Some(evicted) = guard.keys().next().cloned() {
            guard.remove(&evicted);
        }
    }
    guard.insert(key, client.clone());
    Ok(client)
}

fn redirect_location(response: &reqwest::Response, url: &Url) -> CoreResult<Url> {
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            CoreError::InvalidResponse("OCR document redirect missing Location header".to_string())
        })?;
    url.join(location)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR document redirect: {err}")))
}

async fn safe_get_document_url(url: &str) -> CoreResult<(Url, reqwest::Response)> {
    fetch_with_redirects(url, |candidate| async move {
        pin_validated_url(&candidate).await
    })
    .await
}

async fn fetch_with_redirects<P, Fut>(url: &str, resolve: P) -> CoreResult<(Url, reqwest::Response)>
where
    P: Fn(Url) -> Fut,
    Fut: std::future::Future<Output = CoreResult<Vec<SocketAddr>>>,
{
    let mut current_url = Url::parse(url)
        .map_err(|err| CoreError::InvalidRequest(format!("invalid OCR document URL: {err}")))?;

    for _ in 0..MAX_SAFE_FETCH_REDIRECTS {
        let addresses = resolve(current_url.clone()).await?;
        let client = document_fetch_client(&current_url, &addresses)?;
        let response = client
            .get(current_url.clone())
            .send()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        if !response.status().is_redirection() {
            return Ok((current_url, response));
        }
        current_url = redirect_location(&response, &current_url)?;
    }

    Err(CoreError::InvalidRequest(
        "Too many redirects while fetching OCR document URL".to_string(),
    ))
}

fn enforce_download_size(content_length: u64, max_bytes: u64) -> CoreResult<()> {
    if max_bytes == 0 {
        return Err(CoreError::InvalidRequest(
            "OCR document URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0)".to_string(),
        ));
    }
    if content_length > max_bytes {
        let size_mb = content_length as f64 / (1024.0 * 1024.0);
        let max_size_mb = max_bytes as f64 / (1024.0 * 1024.0);
        return Err(CoreError::InvalidRequest(format!(
            "OCR document size ({size_mb:.2}MB) exceeds maximum allowed size ({max_size_mb:.2}MB)"
        )));
    }
    Ok(())
}

async fn read_response_with_limit(
    mut response: reqwest::Response,
    max_bytes: u64,
) -> CoreResult<Vec<u8>> {
    if let Some(content_length) = response.content_length() {
        enforce_download_size(content_length, max_bytes)?;
    } else {
        enforce_download_size(0, max_bytes)?;
    }

    let mut bytes = Vec::new();
    let mut bytes_downloaded: u64 = 0;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?
    {
        bytes_downloaded += chunk.len() as u64;
        enforce_download_size(bytes_downloaded, max_bytes)?;
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

pub(super) async fn convert_document_url_to_data_uri(document: Value) -> CoreResult<Value> {
    let Some((field, url)) = document_url_field(&document)? else {
        return Ok(document);
    };
    if !is_url_requiring_fetch(url) {
        return Ok(document);
    }

    let (_final_url, response) = safe_get_document_url(url).await?;
    let status = response.status();
    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: read_error_body(response).await,
        });
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = read_response_with_limit(response, max_document_download_bytes()).await?;
    let data_uri = format!(
        "data:{content_type};base64,{}",
        BASE64_STANDARD.encode(bytes)
    );

    let mut transformed = document
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    transformed.insert(field.to_string(), Value::String(data_uri));
    Ok(Value::Object(transformed))
}

fn decode_reducto_data_uri(source_url: &str) -> CoreResult<(Vec<u8>, String)> {
    let (header, encoded) = source_url.split_once(',').ok_or_else(|| {
        CoreError::InvalidRequest("Invalid Reducto data URI provided.".to_string())
    })?;
    if !header.contains(";base64") {
        return Err(CoreError::InvalidRequest(
            "Reducto only supports base64-encoded data URIs.".to_string(),
        ));
    }
    let mime = header
        .strip_prefix("data:")
        .and_then(|value| value.split(';').next())
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = BASE64_STANDARD.decode(encoded).map_err(|_| {
        CoreError::InvalidRequest("Invalid Reducto base64 payload provided.".to_string())
    })?;
    Ok((bytes, mime))
}

fn reducto_upload_url(parse_url: &str) -> CoreResult<Url> {
    let mut url = Url::parse(parse_url)
        .map_err(|err| CoreError::InvalidRequest(format!("invalid Reducto parse URL: {err}")))?;
    let path = url.path().trim_end_matches('/');
    let base_path = path
        .strip_suffix("/parse")
        .unwrap_or(path)
        .trim_end_matches('/');
    url.set_path(&format!("{base_path}/upload"));
    url.set_query(None);
    Ok(url)
}

fn reducto_auth_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("authorization"))
        .cloned()
        .collect()
}

async fn upload_reducto_bytes(
    bytes: Vec<u8>,
    mime: String,
    parse_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<String> {
    let upload_url = reducto_upload_url(parse_url)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(&mime)
        .map_err(|err| {
            CoreError::InvalidRequest(format!("invalid Reducto upload MIME type: {err}"))
        })?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let mut request_builder = http_client().post(upload_url).multipart(form);
    for (key, value) in reducto_auth_headers(headers) {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: read_error_body(response).await,
        });
    }
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let response_json: Value = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid Reducto upload response JSON: {err}"))
    })?;
    response_json
        .get("file_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            CoreError::InvalidResponse(format!(
                "Reducto /upload returned 200 without a file_id; got payload={response_json}"
            ))
        })
}

pub(super) async fn upload_reducto_document(
    document: Value,
    parse_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    let Some((field, source_url)) = document_url_field(&document)? else {
        return Err(CoreError::InvalidRequest(
            "Reducto expected OCR preprocessing to produce document_url or image_url".to_string(),
        ));
    };
    if source_url.starts_with("reducto://") {
        return Ok(document);
    }
    if source_url.starts_with("http://") || source_url.starts_with("https://") {
        return Err(CoreError::InvalidRequest(
            "Reducto requires type='file' (auto-uploaded) or a reducto:// id. Plain http(s) URLs are not supported; upload the file first."
                .to_string(),
        ));
    }
    if !source_url.starts_with("data:") {
        return Err(CoreError::InvalidRequest(
            "Reducto requires a reducto:// id or a base64 data URI after OCR preprocessing."
                .to_string(),
        ));
    }

    let (bytes, mime) = decode_reducto_data_uri(source_url)?;
    let file_id = upload_reducto_bytes(bytes, mime, parse_url, headers, timeout).await?;
    let mut transformed = document
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    transformed.insert(field.to_string(), Value::String(file_id));
    Ok(Value::Object(transformed))
}

fn same_origin(left: &str, right: &str) -> bool {
    let Ok(left) = reqwest::Url::parse(left) else {
        return false;
    };
    let Ok(right) = reqwest::Url::parse(right) else {
        return false;
    };
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn retry_after_secs(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2)
}

fn operation_status(response_json: &Value) -> CoreResult<&str> {
    let status = response_json
        .get("status")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("status"))?;
    match status {
        "succeeded" => Ok("succeeded"),
        "running" | "notStarted" => Ok("running"),
        "failed" => {
            let message = response_json
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown error");
            Err(CoreError::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed: {message}"
            )))
        }
        other => Err(CoreError::InvalidResponse(format!(
            "Unknown operation status: {other}"
        ))),
    }
}

pub(super) async fn poll_document_intelligence(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    if !same_origin(operation_url, original_url) {
        return Err(CoreError::InvalidRequest(
            "Azure Document Intelligence: rejected unsafe polling target".to_string(),
        ));
    }

    let start = Instant::now();
    let timeout = timeout.unwrap_or(Duration::from_secs(
        AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS,
    ));
    loop {
        if start.elapsed() > timeout {
            return Err(CoreError::Network(format!(
                "Azure Document Intelligence operation polling timed out after {} seconds",
                timeout.as_secs()
            )));
        }

        let mut request_builder = http_client().get(operation_url);
        for (key, value) in headers {
            if key.eq_ignore_ascii_case("ocp-apim-subscription-key") {
                request_builder = request_builder.header(key, value);
            }
        }
        let response = request_builder
            .send()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        let retry_after = retry_after_secs(&response);
        let status = response.status();
        if !status.is_success() {
            return Err(CoreError::Http {
                status: status.as_u16(),
                body: read_error_body(response).await,
            });
        }
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        let response_json: Value = serde_json::from_str(&text).map_err(|err| {
            CoreError::InvalidResponse(format!("invalid Azure DI poll response JSON: {err}"))
        })?;
        if operation_status(&response_json)? == "succeeded" {
            return Ok(response_json);
        }
        tokio::time::sleep(Duration::from_secs(retry_after)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    struct CountingServer {
        addr: SocketAddr,
        connections: Arc<AtomicUsize>,
        handle: JoinHandle<()>,
    }

    impl CountingServer {
        fn connection_count(&self) -> usize {
            self.connections.load(Ordering::SeqCst)
        }
    }

    impl Drop for CountingServer {
        fn drop(&mut self) {
            self.handle.abort();
        }
    }

    async fn spawn_counting_server(response: Vec<u8>) -> CountingServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let counter = connections.clone();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                let mut discard = [0u8; 2048];
                let _ = socket.read(&mut discard).await;
                let _ = socket.write_all(&response).await;
                let _ = socket.flush().await;
            }
        });
        CountingServer {
            addr,
            connections,
            handle,
        }
    }

    fn http_response(headers: &str, body: &[u8]) -> Vec<u8> {
        [headers.as_bytes(), body].concat()
    }

    #[tokio::test]
    async fn validate_rejects_all_special_use_ranges() {
        let blocked = [
            "http://127.0.0.1/x",
            "http://10.0.0.1/x",
            "http://172.16.0.1/x",
            "http://192.168.1.1/x",
            "http://169.254.169.254/x",
            "http://100.64.0.1/x",
            "http://198.18.0.1/x",
            "http://192.0.2.1/x",
            "http://[::1]/x",
            "http://[fd00::1]/x",
            "http://[fe80::1]/x",
            "http://[::ffff:169.254.169.254]/x",
            "http://[::ffff:10.0.0.1]/x",
            "http://[::ffff:100.64.0.1]/x",
            "http://[::ffff:198.18.0.1]/x",
            "http://[2001:db8::1]/x",
            "http://[100::1]/x",
            "http://[2001::1]/x",
            "http://[2001:2::1]/x",
            "http://[2001:10::1]/x",
            "http://[2001:20::1]/x",
            "http://[2002:c0a8:101::1]/x",
            "http://[2002:a9fe:a9fe::1]/x",
            "http://[64:ff9b:1::1]/x",
            "http://[64:ff9b::192.168.1.1]/x",
            "http://[64:ff9b::169.254.169.254]/x",
            "ftp://8.8.8.8/x",
        ];
        for raw in blocked {
            let url = Url::parse(raw).unwrap();
            let error = pin_validated_url(&url).await.unwrap_err();
            assert!(
                matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
                "{raw} should be rejected, got {error:?}"
            );
        }

        let allowed = Url::parse("http://8.8.8.8/x").unwrap();
        assert_eq!(
            pin_validated_url(&allowed).await.unwrap(),
            vec![SocketAddr::from(([8, 8, 8, 8], 80))]
        );

        let allowed_v6 = [
            "http://[2606:4700:4700::1111]/x",
            "http://[64:ff9b::8.8.8.8]/x",
            "http://[2002:808:808::1]/x",
        ];
        for raw in allowed_v6 {
            let url = Url::parse(raw).unwrap();
            assert!(
                pin_validated_url(&url).await.is_ok(),
                "{raw} should be allowed"
            );
        }
    }

    #[tokio::test]
    async fn loopback_fetch_rejected_without_connecting() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"hi",
        ))
        .await;
        let url = format!("http://127.0.0.1:{}/doc.png", server.addr.port());
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": url,
        }))
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "got {error:?}"
        );
        assert_eq!(server.connection_count(), 0);
    }

    #[tokio::test]
    async fn mapped_ipv6_loopback_fetch_rejected_without_connecting() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"hi",
        ))
        .await;
        let url = format!("http://[::ffff:127.0.0.1]:{}/doc.png", server.addr.port());
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": url,
        }))
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "got {error:?}"
        );
        assert_eq!(server.connection_count(), 0);
    }

    #[tokio::test]
    async fn redirect_to_loopback_rejected_without_connecting_to_target() {
        let target = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
            b"secret",
        ))
        .await;
        let location = format!("http://127.0.0.1:{}/internal", target.addr.port());
        let redirector = spawn_counting_server(
            format!("HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n")
                .into_bytes(),
        )
        .await;
        let redirector_addr = redirector.addr;
        let redirector_port = redirector_addr.port();
        let start_url = format!("http://127.0.0.1:{redirector_port}/doc.png");

        let pin = move |candidate: Url| async move {
            if candidate.port() == Some(redirector_port) {
                return Ok(vec![redirector_addr]);
            }
            pin_validated_url(&candidate).await
        };
        let error = fetch_with_redirects(&start_url, pin).await.unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "got {error:?}"
        );
        assert!(redirector.connection_count() >= 1);
        assert_eq!(target.connection_count(), 0);
    }

    #[tokio::test]
    async fn foreign_operation_location_rejected_without_connecting() {
        let foreign = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"{}",
        ))
        .await;
        let operation_url = format!("http://127.0.0.1:{}/op", foreign.addr.port());
        let original_url = format!("http://127.0.0.1:{}/analyze", foreign.addr.port() ^ 1);

        let error = poll_document_intelligence(&operation_url, &original_url, &[], None)
            .await
            .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("polling target")),
            "got {error:?}"
        );
        assert_eq!(foreign.connection_count(), 0);
    }

    #[tokio::test]
    async fn same_origin_operation_location_polls_target() {
        let body = br#"{"status":"succeeded","analyzeResult":{}}"#;
        let server = spawn_counting_server(http_response(
            &format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()),
            body,
        ))
        .await;
        let operation_url = format!("http://127.0.0.1:{}/op", server.addr.port());

        let result = poll_document_intelligence(&operation_url, &operation_url, &[], None).await;

        assert!(result.is_ok(), "got {result:?}");
        assert!(server.connection_count() >= 1);
    }

    #[tokio::test]
    async fn poll_timeout_override_is_enforced() {
        let body = br#"{"status":"running"}"#;
        let server = spawn_counting_server(http_response(
            &format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\nRetry-After: 0\r\n\r\n",
                body.len()
            ),
            body,
        ))
        .await;
        let operation_url = format!("http://127.0.0.1:{}/op", server.addr.port());

        let error = poll_document_intelligence(
            &operation_url,
            &operation_url,
            &[],
            Some(Duration::from_millis(150)),
        )
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::Network(message) if message.contains("timed out")),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn download_exceeding_cap_without_content_length_is_rejected() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            &vec![b'a'; 4096],
        ))
        .await;
        let url_str = format!("http://127.0.0.1:{}/big", server.addr.port());
        let response = reqwest::Client::new().get(&url_str).send().await.unwrap();
        assert!(response.content_length().is_none());

        let error = read_response_with_limit(response, 1024).await.unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("exceeds maximum")),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn download_within_cap_without_content_length_succeeds() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            &vec![b'a'; 512],
        ))
        .await;
        let url_str = format!("http://127.0.0.1:{}/small", server.addr.port());
        let response = reqwest::Client::new().get(&url_str).send().await.unwrap();
        assert!(response.content_length().is_none());

        let bytes = read_response_with_limit(response, 1024).await.unwrap();

        assert_eq!(bytes.len(), 512);
    }

    #[tokio::test]
    async fn oversized_error_body_without_content_length_is_bounded() {
        let mut raw = b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n".to_vec();
        raw.extend_from_slice(&vec![b'a'; 100_000]);
        let server = spawn_counting_server(raw).await;
        let url_str = format!("http://127.0.0.1:{}/err", server.addr.port());
        let response = reqwest::Client::new().get(&url_str).send().await.unwrap();
        assert!(response.content_length().is_none());
        assert!(!response.status().is_success());

        let body = read_error_body(response).await;

        let prefix = body
            .strip_suffix("... (truncated)")
            .expect("oversized error body must be truncated to the cap");
        assert_eq!(prefix.chars().count(), ERROR_BODY_MAX_CHARS);
        assert!(prefix.chars().all(|c| c == 'a'));
    }

    #[test]
    fn blocks_private_and_metadata_ips() {
        assert!(is_blocked_ip("127.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("100.64.0.1".parse().unwrap()));
        assert!(is_blocked_ip("198.18.0.1".parse().unwrap()));
        assert!(is_blocked_ip("192.0.2.1".parse().unwrap()));
        assert!(is_blocked_ip("::1".parse().unwrap()));
        assert!(is_blocked_ip("fd00::1".parse().unwrap()));
        assert!(is_blocked_ip("fe80::1".parse().unwrap()));
        assert!(is_blocked_ip("fec0::1".parse().unwrap()));
        assert!(is_blocked_ip("2001:db8::1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:100.64.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:198.18.0.1".parse().unwrap()));
        assert!(!is_blocked_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_blocked_ip("::ffff:8.8.8.8".parse().unwrap()));
    }

    #[tokio::test]
    async fn domain_resolving_to_blocked_address_is_rejected_without_connecting() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
            b"secret",
        ))
        .await;
        let start_url = format!("http://localhost:{}/doc", server.addr.port());

        let error = fetch_with_redirects(&start_url, |candidate| async move {
            pin_validated_url(&candidate).await
        })
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "a domain whose DNS answer is a blocked address must be rejected, got {error:?}"
        );
        assert_eq!(server.connection_count(), 0);
    }

    #[tokio::test]
    async fn request_connects_only_to_pinned_address_without_a_second_dns_lookup() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"ok",
        ))
        .await;
        let pinned_addr = server.addr;
        let url = format!("http://pinned.invalid:{}/doc", pinned_addr.port());

        let (_final_url, response) =
            fetch_with_redirects(&url, move |_candidate| async move { Ok(vec![pinned_addr]) })
                .await
                .expect("request must connect via the pinned address");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(server.connection_count(), 1);
    }

    #[tokio::test]
    async fn redirect_across_ports_repins_without_reusing_stale_client() {
        let second = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
            b"second",
        ))
        .await;
        let second_addr = second.addr;
        let first = spawn_counting_server(http_response(
            &format!(
                "HTTP/1.1 302 Found\r\nLocation: http://pinned.invalid:{}/next\r\nContent-Length: 0\r\n\r\n",
                second_addr.port()
            ),
            b"",
        ))
        .await;
        let first_addr = first.addr;
        let start_url = format!("http://pinned.invalid:{}/start", first_addr.port());

        let (final_url, response) = fetch_with_redirects(&start_url, move |candidate| async move {
            match candidate.port() {
                Some(port) if port == first_addr.port() => Ok(vec![first_addr]),
                Some(port) if port == second_addr.port() => Ok(vec![second_addr]),
                _ => Err(blocked_url_error()),
            }
        })
        .await
        .expect("redirect across ports must repin to the second server");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(final_url.port(), Some(second_addr.port()));
        assert_eq!(first.connection_count(), 1);
        assert_eq!(second.connection_count(), 1);
    }

    #[test]
    fn document_client_key_distinguishes_scheme_port_and_addresses() {
        let addr_a = SocketAddr::from(([203, 0, 113, 1], 443));
        let addr_b = SocketAddr::from(([203, 0, 113, 2], 443));
        let base = DocumentClientKey {
            scheme: "https".to_string(),
            host: "docs.example".to_string(),
            port: 443,
            addresses: vec![addr_a],
        };
        let other_scheme = DocumentClientKey {
            scheme: "http".to_string(),
            ..base.clone()
        };
        let other_port = DocumentClientKey {
            port: 8443,
            ..base.clone()
        };
        let other_addrs = DocumentClientKey {
            addresses: vec![addr_a, addr_b],
            ..base.clone()
        };

        assert_ne!(base, other_scheme);
        assert_ne!(base, other_port);
        assert_ne!(base, other_addrs);
        assert_eq!(base, base.clone());
    }

    #[tokio::test]
    async fn convert_document_url_rejects_loopback_fetch() {
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": "http://127.0.0.1/image.png"
        }))
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            CoreError::InvalidRequest(message)
                if message.contains("SSRF protection")
        ));
    }

    #[tokio::test]
    async fn convert_document_url_leaves_data_uri_untouched() {
        let document = json!({
            "type": "image_url",
            "image_url": "data:image/png;base64,abcd"
        });

        let transformed = convert_document_url_to_data_uri(document.clone())
            .await
            .unwrap();

        assert_eq!(transformed, document);
    }
}
