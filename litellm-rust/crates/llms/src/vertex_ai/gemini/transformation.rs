//! Python's `vertex_ai/gemini/transformation.py`: OpenAI messages become Gemini `contents`,
//! the media they point at becomes `inline_data` or `file_data` parts, and the mapped params
//! become the `generateContent` body around them.

use litellm_llms_types::{
    formats::chat_completions::{ChatMessage, ChatMessageContent},
    providers::vertex_ai::GEMINI_FILES_API_URI_PREFIX,
};
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    base_llm::chat::{
        media::{data_url, image_url},
        params::truthy,
    },
    vertex_ai::gemini::vertex_and_google_ai_studio_gemini::{GeminiSurface, is_gemini_3_or_newer},
};

/// `THOUGHT_SIGNATURE_SEPARATOR`: a tool call id carries its thought signature after it.
pub const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

/// `_get_dummy_thought_signature`: base64 of `skip_thought_signature_validator`.
const DUMMY_THOUGHT_SIGNATURE: &str = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=";

/// `GenerationConfig.__annotations__`: the params that travel in `generationConfig`.
const GENERATION_CONFIG_FIELDS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "candidate_count",
    "max_output_tokens",
    "stop_sequences",
    "presence_penalty",
    "frequency_penalty",
    "response_mime_type",
    "response_schema",
    "response_json_schema",
    "responseFormat",
    "seed",
    "responseLogprobs",
    "logprobs",
    "responseModalities",
    "imageConfig",
    "thinkingConfig",
    "mediaResolution",
    "speechConfig",
];

/// `_LITELLM_INTERNAL_EXTRA_BODY_KEYS`.
const INTERNAL_EXTRA_BODY_KEYS: &[&str] = &["cache", "tags"];

/// `FILE_EXTENSIONS` with each type's name and, for the types Gemini accepts
/// (`GEMINI_1_5_ACCEPTED_FILE_TYPES`), its `FILE_MIME_TYPES` entry.
const FILE_TYPES: &[(&str, &str, Option<&str>)] = &[
    ("aac", "AAC", Some("audio/aac")),
    ("csv", "CSV", None),
    ("doc", "DOC", None),
    ("docx", "DOCX", None),
    ("flac", "FLAC", Some("audio/flac")),
    ("flv", "FLV", Some("video/x-flv")),
    ("gif", "GIF", None),
    ("gdoc", "GOOGLE_DOC", None),
    ("gdraw", "GOOGLE_DRAWINGS", None),
    ("gsheet", "GOOGLE_SHEETS", None),
    ("gslides", "GOOGLE_SLIDES", None),
    ("heic", "HEIC", None),
    ("heif", "HEIF", None),
    ("html", "HTML", None),
    ("htm", "HTML", None),
    ("jpeg", "JPEG", Some("image/jpeg")),
    ("jpg", "JPEG", Some("image/jpeg")),
    ("json", "JSON", None),
    ("m4a", "M4A", None),
    ("m4v", "M4V", None),
    ("mov", "MOV", Some("video/quicktime")),
    ("mp3", "MP3", Some("audio/mpeg")),
    ("mp4", "MP4", Some("video/mp4")),
    ("mpeg", "MPEG", Some("video/mpeg")),
    ("mpegps", "MPEGPS", Some("video/mpegps")),
    ("mpg", "MPG", Some("video/mpg")),
    ("mpa", "MPA", Some("audio/m4a")),
    ("mpga", "MPGA", Some("audio/mpga")),
    ("ogg", "OGG", None),
    ("opus", "OPUS", Some("audio/opus")),
    ("pdf", "PDF", Some("application/pdf")),
    ("pcm", "PCM", Some("audio/pcm")),
    ("png", "PNG", Some("image/png")),
    ("ppt", "PPT", None),
    ("pptx", "PPTX", None),
    ("rtf", "RTF", None),
    ("3gpp", "THREE_GPP", Some("video/3gpp")),
    ("txt", "TXT", Some("text/plain")),
    ("wav", "WAV", Some("audio/wav")),
    ("webm", "WEBM", Some("video/webm")),
    ("webp", "WEBP", Some("image/webp")),
    ("wmv", "WMV", Some("video/wmv")),
    ("xls", "XLS", None),
    ("xlsx", "XLSX", None),
];

/// `get_image_mime_type_from_url`, in its table's order.
const URL_MIME_TYPES: &[(&[&str], &str)] = &[
    (&[".jpg", ".jpeg"], "image/jpeg"),
    (&[".png"], "image/png"),
    (&[".webp"], "image/webp"),
    (&[".mp4"], "video/mp4"),
    (&[".mov"], "video/mov"),
    (&[".mpeg", ".mpg"], "video/mpeg"),
    (&[".avi"], "video/avi"),
    (&[".wmv"], "video/wmv"),
    (&[".mpegps"], "video/mpegps"),
    (&[".flv"], "video/flv"),
    (&[".mp3"], "audio/mp3"),
    (&[".wav"], "audio/wav"),
    (&[".ogg"], "audio/ogg"),
    (&[".pdf"], "application/pdf"),
    (&[".txt"], "text/plain"),
];

/// `GEMINI_VIDEO_METADATA_KEYS`.
const VIDEO_METADATA_KEYS: &[(&str, &str)] = &[
    ("fps", "fps"),
    ("start_offset", "startOffset"),
    ("end_offset", "endOffset"),
];

fn bad_request(message: impl Into<String>) -> Error {
    Error::InvalidRequest(ErrorDetail::Message(message.into()))
}

/// `get_image_mime_type_from_url`: the media type a URL's path extension names.
pub fn mime_type_from_url(url: &str) -> Option<&'static str> {
    let lowered = url.to_lowercase();
    let rest = lowered
        .split_once("://")
        .map_or(lowered.as_str(), |(_, rest)| rest);
    let path = rest.find('/').map_or("", |at| &rest[at..]);
    let path = path.split(['?', '#']).next().unwrap_or_default();
    URL_MIME_TYPES
        .iter()
        .find(|(extensions, _)| extensions.iter().any(|ext| path.ends_with(ext)))
        .map(|(_, mime)| *mime)
}

/// `_explicit_mime_type`: `format`, `mime_type` or `content_type`, the first one set.
fn explicit_mime_type(fields: Option<&Map<String, Value>>) -> Option<&str> {
    let fields = fields?;
    ["format", "mime_type", "content_type"]
        .iter()
        .find_map(|key| fields.get(*key).filter(|value| truthy(value)))
        .and_then(Value::as_str)
}

/// `_apply_gemini_mime_type_aliases`.
fn mime_type_alias(mime_type: &str) -> String {
    let normalized = mime_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    match normalized.as_str() {
        "image/jpg" => "image/jpeg".to_owned(),
        _ => normalized,
    }
}

/// `os.path.splitext(url)[1][1:]`.
fn extension(url: &str) -> &str {
    let base = url.rsplit('/').next().unwrap_or(url);
    let base = base.trim_start_matches('.');
    base.rfind('.').map_or("", |at| &base[at + 1..])
}

/// The media type Gemini takes a `gs://` object of this extension as.
fn gcs_mime_type(extension: &str) -> Result<&'static str, Error> {
    let lowered = extension.to_lowercase();
    let (_, name, mime) = FILE_TYPES
        .iter()
        .find(|(ext, _, _)| *ext == lowered)
        .ok_or_else(|| bad_request(format!("Unknown file type for extension: {extension}")))?;
    mime.ok_or_else(|| {
        bad_request(format!(
            "File type not supported by gemini - FileType.{name}"
        ))
    })
}

/// `_convert_detail_to_media_resolution_enum`.
fn media_resolution(detail: Option<&Value>) -> Option<&'static str> {
    match detail?.as_str()? {
        "low" => Some("MEDIA_RESOLUTION_LOW"),
        "medium" => Some("MEDIA_RESOLUTION_MEDIUM"),
        "high" => Some("MEDIA_RESOLUTION_HIGH"),
        "ultra_high" => Some("MEDIA_RESOLUTION_ULTRA_HIGH"),
        _ => None,
    }
}

fn resolution_priority(detail: &str) -> u8 {
    match detail {
        "ultra_high" => 4,
        "high" => 3,
        "medium" => 2,
        "low" => 1,
        _ => 0,
    }
}

/// A `data:` URL as `{data, mime_type}`, the way `convert_to_anthropic_image_obj` reads it. A
/// remote URL here was not downloaded first, which only the host can do.
fn inline_blob(url: &str, format: Option<&str>) -> Result<Value, Error> {
    if url.starts_with("http") {
        return Err(Error::Unsupported("media the host did not download"));
    }
    let (media_type, data) = data_url(url).ok_or_else(|| {
        bad_request(format!(
            "Image url not in expected format. Example Expected input - \"image_url\": \"data:image/jpeg;base64,{{base64_image}}\". Received: {}",
            url.chars().take(64).collect::<String>()
        ))
    })?;
    let media_type = match format {
        Some(format) => format.to_owned(),
        None => media_type.replace("\\/", "/"),
    };
    Ok(json!({"data": data, "mime_type": media_type}))
}

/// `_process_gemini_media` with `_apply_gemini_metadata`.
fn media_part(
    url: &str,
    format: Option<&str>,
    resolution: Option<&'static str>,
    video_metadata: Option<&Value>,
    model: &str,
) -> Result<Value, Error> {
    let mut part = if url.contains("gs://") {
        let mime_type = match format {
            Some(format) => mime_type_alias(format),
            None => match extension(url) {
                "" => {
                    return Err(Error::Unsupported(
                        "a gs:// URI with no file extension and no format",
                    ));
                }
                extension => gcs_mime_type(extension)?.to_owned(),
            },
        };
        json!({"file_data": {"mime_type": mime_type, "file_uri": url}})
    } else if url.starts_with(GEMINI_FILES_API_URI_PREFIX) {
        match format {
            Some(format) => json!({"file_data": {"mime_type": format, "file_uri": url}}),
            None => json!({"file_data": {"file_uri": url}}),
        }
    } else if let Some(mime_type) = url
        .contains("https://")
        .then(|| format.or_else(|| mime_type_from_url(url)))
        .flatten()
    {
        json!({"file_data": {"mime_type": mime_type, "file_uri": url}})
    } else if url.contains("http://") || url.contains("https://") || url.contains("base64") {
        json!({"inline_data": inline_blob(url, format)?})
    } else {
        return Err(bad_request(format!("Invalid image received - {url}")));
    };
    if let Some(level) = resolution
        && is_gemini_3_or_newer(model)
    {
        part["media_resolution"] = json!({"level": level});
    }
    if let Some(Value::Object(metadata)) = video_metadata {
        let mapped: Map<String, Value> = VIDEO_METADATA_KEYS
            .iter()
            .filter_map(|(openai, gemini)| {
                metadata
                    .get(*openai)
                    .map(|value| ((*gemini).to_owned(), value.clone()))
            })
            .collect();
        if !mapped.is_empty() {
            part["video_metadata"] = Value::Object(mapped);
        }
    }
    Ok(part)
}

/// The `image_url` element of a user message.
fn image_url_part(element: &Value, model: &str) -> Result<Value, Error> {
    let (url, format, resolution) = match element.get("image_url") {
        None | Some(Value::Null) => {
            return Err(bad_request(
                "Invalid message content: element type is 'image_url' but 'image_url' field is missing ",
            ));
        }
        Some(Value::Object(image)) => {
            let url = image.get("url").and_then(Value::as_str).ok_or_else(|| {
                bad_request(
                    "Invalid message content: element type is 'image_url' but 'url' field is missing inside 'image_url' ",
                )
            })?;
            (
                url,
                explicit_mime_type(Some(image)),
                media_resolution(image.get("detail")),
            )
        }
        Some(Value::String(url)) => (url.as_str(), None, None),
        Some(other) => {
            return Err(bad_request(format!(
                "Invalid message content: image_url must be a string or an object, got {other}"
            )));
        }
    };
    media_part(url, format, resolution, None, model)
}

/// The `file` element of a user message.
fn file_part(element: &Value, model: &str) -> Result<Value, Error> {
    let Some(Value::Object(file)) = element.get("file") else {
        return Err(bad_request(
            "Content block has type='file' but is missing the required 'file' field",
        ));
    };
    let source = ["file_id", "file_data"]
        .iter()
        .find_map(|key| file.get(*key).filter(|value| truthy(value)))
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request("Unknown file type. Please pass in a file_id or file_data"))?;
    media_part(
        source,
        explicit_mime_type(Some(file)),
        media_resolution(file.get("detail")),
        file.get("video_metadata"),
        model,
    )
}

/// The `input_audio` element of a user message, which Gemini takes inline.
fn audio_part(element: &Value, model: &str) -> Result<Option<Value>, Error> {
    let audio = element.get("input_audio");
    let data = audio
        .and_then(|audio| audio.get("data"))
        .and_then(Value::as_str);
    let format = audio
        .and_then(|audio| audio.get("format"))
        .and_then(Value::as_str);
    let (Some(data), Some(format)) = (data, format) else {
        return Ok(None);
    };
    let format = match format.starts_with("audio/") {
        true => format.to_owned(),
        false => format!("audio/{format}"),
    };
    media_part(
        &format!("data:{format};base64,{data}"),
        Some(&format),
        None,
        None,
        model,
    )
    .map(Some)
}

/// The parts of one user or (merged) system message.
fn user_parts(message: &Value, model: &str) -> Result<Vec<Value>, Error> {
    match message.get("content") {
        Some(Value::String(text)) => Ok(vec![json!({"text": text})]),
        Some(Value::Array(elements)) => {
            let mut parts = Vec::new();
            for element in elements {
                match element.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = element.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            parts.push(json!({"text": text}));
                        }
                    }
                    Some("image_url") => parts.push(image_url_part(element, model)?),
                    Some("input_audio") => parts.extend(audio_part(element, model)?),
                    Some("file") => parts.push(file_part(element, model)?),
                    _ => {}
                }
            }
            Ok(parts)
        }
        _ => Ok(Vec::new()),
    }
}

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn provider_signature(holder: &Value) -> Option<&str> {
    holder
        .get("provider_specific_fields")
        .and_then(|fields| string_field(fields, "thought_signature"))
}

/// `get_thought_signature_from_tool`: the tool's, its function's, or the one after the id's
/// separator.
pub fn thought_signature_from_tool(tool: &Value) -> Option<&str> {
    provider_signature(tool)
        .or_else(|| tool.get("function").and_then(provider_signature))
        .or_else(|| {
            tool.get("id")
                .and_then(Value::as_str)
                .and_then(|id| id.split_once(THOUGHT_SIGNATURE_SEPARATOR))
                .map(|(_, signature)| signature)
                .filter(|signature| !signature.is_empty())
        })
}

/// `_collect_tool_call_thought_signatures`.
fn tool_call_signatures(message: &Value) -> Vec<&str> {
    let mut signatures: Vec<&str> = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(thought_signature_from_tool)
        .collect();
    if let Some(function_call) = message.get("function_call").filter(|f| f.is_object()) {
        signatures.extend(provider_signature(function_call));
    }
    let invocations = message
        .get("provider_specific_fields")
        .and_then(|fields| fields.get("server_side_tool_invocations"))
        .and_then(Value::as_array);
    for invocation in invocations.into_iter().flatten() {
        for key in ["thought_signature", "response_thought_signature"] {
            signatures.extend(string_field(invocation, key));
        }
    }
    signatures
}

/// `_gemini_tool_call_invoke_helper`.
fn function_call(function: &Value, id: Option<&str>) -> Result<Value, Error> {
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = match function.get("arguments") {
        None => json!({"type": "object"}),
        Some(Value::String(arguments)) if arguments.is_empty() => json!({"type": "object"}),
        Some(Value::String(arguments)) => serde_json::from_str(arguments).map_err(|error| {
            bad_request(format!(
                "Unable to convert openai tool calls to gemini tool calls. Received error={error}"
            ))
        })?,
        Some(other) => {
            return Err(bad_request(format!(
                "Unable to convert openai tool calls to gemini tool calls. Received arguments={other}"
            )));
        }
    };
    let mut call = json!({"name": name, "args": args});
    if let Some(id) = id
        .map(|id| id.split(THOUGHT_SIGNATURE_SEPARATOR).next().unwrap_or(id))
        .filter(|id| !id.is_empty())
    {
        call["id"] = json!(id);
    }
    Ok(call)
}

/// `convert_to_gemini_tool_call_invoke`. Gemini 3 and newer sign the first call of a turn, so
/// history from an older model gets Google's placeholder signature there.
fn tool_call_parts(message: &Value, gemini_3: bool) -> Result<Vec<Value>, Error> {
    let mut parts: Vec<Value> = Vec::new();
    if let Some(Value::Array(calls)) = message.get("tool_calls") {
        for tool in calls {
            let Some(function) = tool.get("function") else {
                continue;
            };
            let id = gemini_3
                .then(|| tool.get("id").and_then(Value::as_str))
                .flatten();
            let mut part = json!({"function_call": function_call(function, id)?});
            let signature = thought_signature_from_tool(tool)
                .or((parts.is_empty() && gemini_3).then_some(DUMMY_THOUGHT_SIGNATURE));
            if let Some(signature) = signature {
                part["thoughtSignature"] = json!(signature);
            }
            parts.push(part);
        }
    } else if let Some(function) = message.get("function_call").filter(|f| !f.is_null()) {
        let mut part = json!({"function_call": function_call(function, None)?});
        let signature =
            provider_signature(function).or(gemini_3.then_some(DUMMY_THOUGHT_SIGNATURE));
        if let Some(signature) = signature {
            part["thoughtSignature"] = json!(signature);
        }
        parts.push(part);
    }
    Ok(parts)
}

/// `snake_to_camel`.
pub fn snake_to_camel(name: &str) -> String {
    let mut pieces = name.split('_');
    let mut camel = pieces.next().unwrap_or_default().to_owned();
    for piece in pieces {
        let mut chars = piece.chars();
        if let Some(first) = chars.next() {
            camel.extend(first.to_uppercase());
            camel.push_str(&chars.as_str().to_lowercase());
        }
    }
    camel
}

/// `camel_to_snake`.
pub fn camel_to_snake(name: &str) -> String {
    let mut snake = String::with_capacity(name.len() + 4);
    for (at, ch) in name.chars().enumerate() {
        if at > 0 && ch.is_uppercase() {
            snake.push('_');
        }
        snake.extend(ch.to_lowercase());
    }
    snake
}

/// `_get_equivalent_key`.
fn equivalent_key<'a>(key: &str, keys: impl Fn(&str) -> Option<&'a str>) -> Option<&'a str> {
    keys(key)
        .or_else(|| keys(&snake_to_camel(key)))
        .or_else(|| keys(&camel_to_snake(key)))
}

/// `check_if_part_exists_in_parts`, with `thoughtSignature` left out of the comparison.
fn part_exists(parts: &[Value], part: &Value) -> bool {
    let Some(part) = part.as_object() else {
        return false;
    };
    parts.iter().filter_map(Value::as_object).any(|existing| {
        part.iter()
            .filter(|(key, _)| key.as_str() != "thoughtSignature")
            .all(|(key, value)| {
                equivalent_key(key, |name| {
                    existing.get_key_value(name).map(|(key, _)| key.as_str())
                })
                .is_some_and(|found| existing.get(found) == Some(value))
            })
    })
}

/// The `toolCall` and `toolResponse` parts Gemini ran server side, put back for the next turn.
fn server_side_parts(message: &Value) -> Vec<Value> {
    let invocations = message
        .get("provider_specific_fields")
        .and_then(|fields| fields.get("server_side_tool_invocations"))
        .and_then(Value::as_array);
    let mut parts = Vec::new();
    for invocation in invocations.into_iter().flatten().filter(|i| i.is_object()) {
        let field = |key: &str| invocation.get(key).cloned().unwrap_or(Value::Null);
        let mut call = json!({"toolCall": {"toolType": field("tool_type"), "id": field("id"), "args": field("args")}});
        if let Some(signature) = invocation.get("thought_signature") {
            call["thoughtSignature"] = signature.clone();
        }
        parts.push(call);
        if let Some(response) = invocation.get("response") {
            let mut tool_response = json!({"id": field("id"), "response": response});
            if let Some(tool_type) = invocation.get("tool_type").filter(|t| truthy(t)) {
                tool_response["toolType"] = tool_type.clone();
            }
            let mut part = json!({"toolResponse": tool_response});
            if let Some(signature) = invocation.get("response_thought_signature") {
                part["thoughtSignature"] = signature.clone();
            }
            parts.push(part);
        }
    }
    parts
}

/// The parts of one assistant message.
fn assistant_parts(
    message: &Value,
    parts: &mut Vec<Value>,
    model: &str,
    gemini_3: bool,
) -> Result<(), Error> {
    if let Some(reasoning) = message.get("reasoning_content").filter(|r| !r.is_null()) {
        parts.push(json!({"thought": true, "text": reasoning}));
    }
    match message.get("content") {
        Some(Value::Array(elements)) => parts.extend(
            elements
                .iter()
                .filter(|element| element.get("type").and_then(Value::as_str) == Some("text"))
                .map(
                    |element| json!({"text": element.get("text").cloned().unwrap_or(Value::Null)}),
                ),
        ),
        Some(Value::String(text)) => {
            let signature = message
                .get("provider_specific_fields")
                .and_then(|fields| fields.get("thought_signatures"))
                .and_then(Value::as_array)
                .and_then(|signatures| signatures.first())
                .filter(|first| {
                    first
                        .as_str()
                        .is_none_or(|first| !tool_call_signatures(message).contains(&first))
                });
            parts.push(match signature {
                Some(signature) => json!({"text": text, "thoughtSignature": signature}),
                None => json!({"text": text}),
            });
        }
        _ => {}
    }
    for image in message
        .get("images")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(Value::Object(image_url)) = image.get("image_url") else {
            continue;
        };
        if let Some(url) = image_url
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
        {
            parts.push(media_part(
                url,
                explicit_mime_type(Some(image_url)),
                media_resolution(image_url.get("detail")),
                None,
                model,
            )?);
        }
    }
    for part in tool_call_parts(message, gemini_3)? {
        if !part_exists(parts, &part) {
            parts.push(part);
        }
    }
    parts.extend(server_side_parts(message));
    Ok(())
}

/// `_contains_part_reference`: Gemini reads `{"$ref": "<non-empty>"}` in a function response
/// as a pointer to one of its parts, except under a list nested directly in a list.
fn contains_part_reference(node: &Value) -> bool {
    match node {
        Value::Object(object) => {
            object
                .get("$ref")
                .and_then(Value::as_str)
                .is_some_and(|reference| !reference.is_empty())
                || object.values().any(contains_part_reference)
        }
        Value::Array(items) => items
            .iter()
            .filter(|item| !item.is_array())
            .any(contains_part_reference),
        _ => false,
    }
}

/// `_may_spell_ref_key`.
fn may_spell_ref_key(raw: &str) -> bool {
    raw.contains("\"$ref\"")
        || ["\\u0024", "\\u0072", "\\u0065", "\\u0066"]
            .iter()
            .any(|escape| raw.contains(escape))
}

/// The media of a tool result, inline the way `convert_to_gemini_tool_call_result` puts it.
fn tool_result_blob(content: &Value) -> Result<Option<Value>, Error> {
    match content.get("type").and_then(Value::as_str) {
        Some("image") => {
            let source = content.get("source");
            if source.and_then(|s| s.get("type")).and_then(Value::as_str) != Some("base64") {
                return Ok(None);
            }
            let field = |key: &str, default: &str| {
                source
                    .and_then(|s| s.get(key))
                    .cloned()
                    .unwrap_or_else(|| json!(default))
            };
            Ok(Some(
                json!({"data": field("data", ""), "mime_type": field("media_type", "image/jpeg")}),
            ))
        }
        Some("input_image" | "image_url") => {
            let url = match content.get("image_url") {
                Some(Value::Object(image)) => image.get("url").and_then(Value::as_str),
                Some(Value::String(url)) => Some(url.as_str()),
                _ => None,
            };
            match url.filter(|url| !url.is_empty()) {
                Some(url) => tool_result_inline(url),
                None => Ok(None),
            }
        }
        Some("file" | "input_file") => {
            let data = content
                .get("file_data")
                .filter(|data| truthy(data))
                .or_else(|| match content.get("file") {
                    Some(Value::Object(file)) => file.get("file_data"),
                    Some(file @ Value::String(_)) => Some(file),
                    _ => None,
                })
                .and_then(Value::as_str)
                .filter(|data| !data.is_empty());
            match data {
                Some(data) => tool_result_inline(data),
                None => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Python fetches a remote tool result image while it converts the message and skips one it
/// cannot read; only the host can fetch, so a remote one it left is refused.
fn tool_result_inline(url: &str) -> Result<Option<Value>, Error> {
    if url.starts_with("http") {
        return Err(Error::Unsupported("remote media in a tool result"));
    }
    Ok(inline_blob(url, None).ok())
}

/// `convert_to_gemini_tool_call_result`.
fn tool_result_part(
    message: &Value,
    last_tool_calls: Option<&Value>,
    gemini_3: bool,
) -> Result<Value, Error> {
    let mut text = String::new();
    let mut blobs: Vec<Value> = Vec::new();
    match message.get("content") {
        Some(Value::String(content)) => {
            text = content.clone();
            if content.len() >= 5
                && content.is_char_boundary(5)
                && content[..5].eq_ignore_ascii_case("data:")
                && let Some((mime, data)) = content[5..].split_once(";base64,")
                && mime.starts_with("image/")
            {
                let mime = mime.split(';').next().unwrap_or(mime).trim();
                blobs.push(json!({"data": data, "mime_type": mime}));
                text.clear();
            }
        }
        Some(Value::Array(contents)) => {
            for content in contents {
                if content.get("type").and_then(Value::as_str) == Some("text") {
                    text.push_str(
                        content
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    );
                } else {
                    blobs.extend(tool_result_blob(content)?);
                }
            }
        }
        _ => {}
    }
    let mut name = message
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let tool_call_id = message
        .get("tool_call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    if let Some(calls) = last_tool_calls
        .and_then(|last| last.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for call in calls {
            if let Some(id) = tool_call_id
                && call.get("id").and_then(Value::as_str) == Some(id)
            {
                name = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
            }
        }
    }
    if name.is_empty() {
        return Err(bad_request(format!(
            "Missing corresponding tool call for tool response message. Received - message={message}"
        )));
    }
    let trimmed = text.trim();
    let response = match trimmed.starts_with('{') || trimmed.starts_with('[') {
        true => match serde_json::from_str::<Value>(&text) {
            Ok(parsed @ Value::Object(_))
                if !(may_spell_ref_key(&text) && contains_part_reference(&parsed)) =>
            {
                parsed
            }
            _ => json!({"content": text}),
        },
        false => json!({"content": text}),
    };
    let mut function_response = json!({"name": name, "response": response});
    if gemini_3
        && let Some(id) = tool_call_id
            .and_then(|id| id.split(THOUGHT_SIGNATURE_SEPARATOR).next())
            .filter(|id| !id.is_empty())
    {
        function_response["id"] = json!(id);
    }
    if !blobs.is_empty() {
        function_response["parts"] = Value::Array(
            blobs
                .into_iter()
                .map(|blob| json!({"inline_data": blob}))
                .collect(),
        );
    }
    Ok(json!({"function_response": function_response}))
}

fn role(message: &Value) -> &str {
    message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// `gemini_convert_messages_with_history`: consecutive user (and unextracted system) messages
/// merge into one `user` turn, consecutive assistant messages into one `model` turn, and
/// consecutive tool results into one `user` turn of function responses.
pub fn convert_messages(messages: &[Value], model: &str) -> Result<Vec<Value>, Error> {
    let gemini_3 = is_gemini_3_or_newer(model);
    let mut contents: Vec<Value> = Vec::new();
    let mut last_tool_calls: Option<&Value> = None;
    let mut tool_responses: Vec<Value> = Vec::new();
    let mut at = 0;
    while at < messages.len() {
        let start = at;
        let mut user: Vec<Value> = Vec::new();
        while at < messages.len() && matches!(role(&messages[at]), "user" | "system") {
            user.extend(user_parts(&messages[at], model)?);
            at += 1;
        }
        if !user.is_empty() {
            if !user
                .iter()
                .any(|part| part.get("text").is_some_and(|text| !text.is_null()))
            {
                user.push(json!({"text": " "}));
            }
            contents.push(json!({"role": "user", "parts": user}));
        }
        let mut assistant: Vec<Value> = Vec::new();
        while at < messages.len() && role(&messages[at]) == "assistant" {
            let message = &messages[at];
            assistant_parts(message, &mut assistant, model, gemini_3)?;
            if message.get("tool_calls").is_some_and(truthy)
                || message.get("function_call").is_some_and(|f| !f.is_null())
            {
                last_tool_calls = Some(message);
            }
            at += 1;
        }
        if !assistant.is_empty() {
            contents.push(json!({"role": "model", "parts": assistant}));
        }
        let is_tool_result = |message: &Value| matches!(role(message), "tool" | "function");
        if at < messages.len() && is_tool_result(&messages[at]) {
            tool_responses.push(tool_result_part(&messages[at], last_tool_calls, gemini_3)?);
            at += 1;
        }
        if at < messages.len() && !is_tool_result(&messages[at]) && !tool_responses.is_empty() {
            contents.push(json!({"role": "user", "parts": std::mem::take(&mut tool_responses)}));
        }
        if at == start {
            return Err(bad_request(format!(
                "Invalid Message passed in - {}",
                messages[at]
            )));
        }
    }
    if !tool_responses.is_empty() {
        contents.push(json!({"role": "user", "parts": tool_responses}));
    }
    if contents.is_empty() {
        contents.push(json!({"role": "user", "parts": [{"text": " "}]}));
    }
    Ok(contents)
}

/// `transform_system_message`: when the model takes a system instruction, the system messages
/// leave the conversation for it, and a conversation left empty gets a `.` user turn.
pub fn transform_system_message(
    supports_system_message: bool,
    messages: Vec<Value>,
) -> (Option<Value>, Vec<Value>) {
    if !supports_system_message {
        return (None, messages);
    }
    let mut blocks: Vec<Value> = Vec::new();
    let mut rest: Vec<Value> = Vec::new();
    for message in messages {
        let block = match (role(&message), message.get("content")) {
            ("system", Some(Value::String(text))) => Some(json!({"text": text})),
            ("system", Some(Value::Array(items))) => Some(json!({"text": items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<String>()})),
            _ => None,
        };
        match block {
            Some(block) => blocks.push(block),
            None => rest.push(message),
        }
    }
    if blocks.is_empty() {
        return (None, rest);
    }
    if rest.is_empty() {
        rest.push(json!({"role": "user", "content": "."}));
    }
    (Some(json!({"parts": blocks})), rest)
}

/// `is_cached_message`: an ephemeral `cache_control` on the message or a text part, which
/// Python turns into a Gemini context cache it creates first.
pub fn is_cached_message(message: &Value) -> bool {
    let ephemeral = |holder: &Value| {
        holder
            .get("cache_control")
            .and_then(|control| control.get("type"))
            .and_then(Value::as_str)
            == Some("ephemeral")
    };
    ephemeral(message)
        || message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts.iter().any(|part| {
                    part.get("type").and_then(Value::as_str) == Some("text") && ephemeral(part)
                })
            })
}

/// `_extract_max_media_resolution_from_messages`.
fn max_media_resolution(messages: &[Value]) -> Option<&'static str> {
    let mut best: Option<&str> = None;
    for part in messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
    {
        let holder = match part.get("type").and_then(Value::as_str) {
            Some("image_url") => part.get("image_url"),
            Some("file") => part.get("file"),
            _ => None,
        };
        let detail = holder
            .filter(|holder| holder.is_object())
            .and_then(|holder| holder.get("detail"))
            .and_then(Value::as_str)
            .filter(|detail| !detail.is_empty());
        if let Some(detail) = detail
            && resolution_priority(detail) > best.map_or(0, resolution_priority)
        {
            best = Some(detail);
        }
    }
    media_resolution(best.map(|detail| json!(detail)).as_ref())
}

/// `_rewrite_google_maps_response_format`: Google Maps grounding rejects
/// `response_mime_type: application/json`, so the newer `responseFormat` carries it.
fn rewrite_google_maps_response_format(body: &mut Map<String, Value>) {
    let has_maps = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(|tool| tool.get("googleMaps").is_some()));
    let Some(Value::Object(config)) = body.get_mut("generationConfig") else {
        return;
    };
    if !has_maps
        || config.get("response_mime_type").and_then(Value::as_str) != Some("application/json")
    {
        return;
    }
    let schema = config
        .remove("response_json_schema")
        .or_else(|| config.remove("response_schema"));
    config.remove("response_mime_type");
    let mut text = json!({"mimeType": "APPLICATION_JSON"});
    if let Some(schema) = schema {
        text["schema"] = schema;
    }
    config.insert("responseFormat".into(), json!({"text": text}));
}

/// `transform_request_body`: the `generateContent` body for messages already stripped of the
/// system instruction and params already mapped.
pub fn transform_request_body(
    messages: &[Value],
    system_instruction: Option<Value>,
    model: &str,
    mut params: Map<String, Value>,
    surface: GeminiSurface,
) -> Result<Value, Error> {
    let contents = convert_messages(messages, model)?;
    let tools = params.remove("tools");
    let tool_choice = params.remove("tool_choice");
    let include_server_side_tool_invocations = params
        .remove("include_server_side_tool_invocations")
        .is_some_and(|flag| truthy(&flag));
    let safety_settings = params.remove("safety_settings");
    params.remove("output_config");
    let labels: Option<Map<String, Value>> = match params.remove("labels") {
        Some(Value::Object(labels)) => Some(
            labels
                .into_iter()
                .filter(|(_, value)| value.is_string())
                .collect(),
        ),
        _ => None,
    };
    let cached_content = params.remove("cached_content");
    let service_tier = params.remove("service_tier").filter(truthy);
    let continuation_token = params.remove("continuation_token").filter(truthy);
    let extra_body = params.remove("extra_body");

    let mut generation_config: Map<String, Value> = params
        .into_iter()
        .filter(|(key, _)| {
            equivalent_key(key, |name| {
                GENERATION_CONFIG_FIELDS
                    .iter()
                    .find(|field| **field == name)
                    .copied()
            })
            .is_some()
        })
        .collect();
    if model.contains("gemini-2")
        && let Some(level) = max_media_resolution(messages)
    {
        generation_config.insert("mediaResolution".into(), json!(level));
    }

    let mut body = Map::new();
    body.insert("contents".into(), Value::Array(contents));
    if let Some(system_instruction) = system_instruction {
        body.insert("system_instruction".into(), system_instruction);
    }
    if let Some(tools) = tools.filter(|tools| !tools.is_null()) {
        body.insert("tools".into(), tools);
    }
    if let Some(tool_choice) = tool_choice.filter(|choice| !choice.is_null()) {
        body.insert("toolConfig".into(), tool_choice);
    }
    if include_server_side_tool_invocations {
        let config = body
            .entry("toolConfig")
            .or_insert_with(|| Value::Object(Map::new()));
        config["includeServerSideToolInvocations"] = json!(true);
    }
    if let Some(safety_settings) = safety_settings.filter(|s| !s.is_null()) {
        body.insert("safetySettings".into(), safety_settings);
    }
    if !generation_config.is_empty() {
        body.insert("generationConfig".into(), Value::Object(generation_config));
    }
    if let Some(cached_content) = cached_content.filter(|c| !c.is_null()) {
        body.insert("cachedContent".into(), cached_content);
    }
    if let Some(service_tier) = service_tier {
        let tier = match service_tier.as_str().map(str::to_lowercase) {
            Some(tier) if tier == "default" => json!("standard"),
            Some(tier) => json!(tier),
            None => service_tier,
        };
        body.insert("serviceTier".into(), tier);
    }
    if let Some(token) = continuation_token {
        body.insert("continuationToken".into(), token);
    }
    if let Some(labels) = labels.filter(|labels| !labels.is_empty())
        && surface == GeminiSurface::Vertex
    {
        body.insert("labels".into(), Value::Object(labels));
    }
    if let Some(Value::Object(extra)) = extra_body {
        for (key, value) in extra {
            if INTERNAL_EXTRA_BODY_KEYS.contains(&key.as_str()) {
                continue;
            }
            match (body.get_mut(&key), value) {
                (Some(Value::Object(existing)), Value::Object(update)) => existing.extend(update),
                (_, value) => {
                    body.insert(key, value);
                }
            }
        }
    }
    rewrite_google_maps_response_format(&mut body);
    Ok(Value::Object(body))
}

/// The media Python downloads while it builds the request (`should_inline_remote_media`, and
/// the images a tool result names, which `convert_to_gemini_tool_call_result` fetches). Google
/// AI Studio cannot read a URL; Vertex AI reads an `https` one whose media type is known.
pub fn remote_media(messages: &[ChatMessage], surface: GeminiSurface) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    for message in messages {
        let Some(ChatMessageContent::Parts(parts)) = &message.content else {
            continue;
        };
        for part in parts {
            let kind = part.get("type").and_then(Value::as_str);
            let (url, fields) = match kind {
                Some("image_url") => (image_url(part), part.get("image_url")),
                Some("file") => (
                    part.get("file")
                        .and_then(|file| file.get("file_id"))
                        .and_then(Value::as_str),
                    part.get("file"),
                ),
                _ => continue,
            };
            let Some(url) =
                url.filter(|url| url.starts_with("http://") || url.starts_with("https://"))
            else {
                continue;
            };
            let by_reference = url.starts_with(GEMINI_FILES_API_URI_PREFIX);
            let inlined = !by_reference
                && match surface {
                    GeminiSurface::AiStudio => true,
                    GeminiSurface::Vertex => {
                        url.starts_with("http://")
                            || (explicit_mime_type(fields.and_then(Value::as_object)).is_none()
                                && mime_type_from_url(url).is_none())
                    }
                };
            let tool_image = message.role == "tool" && kind == Some("image_url");
            if (inlined || tool_image) && !urls.iter().any(|seen| seen == url) {
                urls.push(url.to_owned());
            }
        }
    }
    urls
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn messages(value: Value) -> Vec<Value> {
        let Value::Array(messages) = value else {
            panic!("messages are a list");
        };
        messages
    }

    #[rstest]
    #[case::user_and_system_merge(
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]),
        json!([{"role": "user", "parts": [{"text": "be brief"}, {"text": "hi"}]}])
    )]
    #[case::consecutive_assistants_merge(
        json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "assistant", "content": "c"}]),
        json!([{"role": "user", "parts": [{"text": "a"}]}, {"role": "model", "parts": [{"text": "b"}, {"text": "c"}]}])
    )]
    #[case::empty_conversation(json!([]), json!([{"role": "user", "parts": [{"text": " "}]}]))]
    #[case::image_only_gets_blank_text(
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]),
        json!([{"role": "user", "parts": [{"inline_data": {"data": "AAAA", "mime_type": "image/png"}}, {"text": " "}]}])
    )]
    #[case::reasoning_becomes_a_thought(
        json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "think"}]),
        json!([{"role": "user", "parts": [{"text": "q"}]}, {"role": "model", "parts": [{"thought": true, "text": "think"}, {"text": "a"}]}])
    )]
    #[case::text_signature_replayed(
        json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "provider_specific_fields": {"thought_signatures": ["sig"]}}]),
        json!([{"role": "user", "parts": [{"text": "q"}]}, {"role": "model", "parts": [{"text": "a", "thoughtSignature": "sig"}]}])
    )]
    fn converts_messages_like_python(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(
            Value::Array(convert_messages(&messages(input), "gemini-2.5-flash").unwrap()),
            expected
        );
    }

    fn tool_conversation() -> Value {
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": ""}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
            {"role": "tool", "tool_call_id": "call_2", "content": "noon"},
            {"role": "user", "content": "thanks"}
        ])
    }

    #[rstest]
    #[case::older_model("gemini-2.5-flash", json!([
        {"role": "user", "parts": [{"text": "weather?"}]},
        {"role": "model", "parts": [
            {"function_call": {"name": "get_weather", "args": {"city": "Paris"}}},
            {"function_call": {"name": "get_time", "args": {"type": "object"}}}
        ]},
        {"role": "user", "parts": [
            {"function_response": {"name": "get_weather", "response": {"temp": 21}}},
            {"function_response": {"name": "get_time", "response": {"content": "noon"}}}
        ]},
        {"role": "user", "parts": [{"text": "thanks"}]}
    ]))]
    #[case::gemini_3_forwards_ids_and_signs_the_first_call("gemini-3-pro-preview", json!([
        {"role": "user", "parts": [{"text": "weather?"}]},
        {"role": "model", "parts": [
            {"function_call": {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}, "thoughtSignature": DUMMY_THOUGHT_SIGNATURE},
            {"function_call": {"name": "get_time", "args": {"type": "object"}, "id": "call_2"}}
        ]},
        {"role": "user", "parts": [
            {"function_response": {"name": "get_weather", "response": {"temp": 21}, "id": "call_1"}},
            {"function_response": {"name": "get_time", "response": {"content": "noon"}, "id": "call_2"}}
        ]},
        {"role": "user", "parts": [{"text": "thanks"}]}
    ]))]
    fn tool_calls_and_results_round_trip(#[case] model: &str, #[case] expected: Value) {
        assert_eq!(
            Value::Array(convert_messages(&messages(tool_conversation()), model).unwrap()),
            expected
        );
    }

    #[rstest]
    #[case::from_provider_fields(json!({"id": "call_1", "provider_specific_fields": {"thought_signature": "a"}, "function": {}}), Some("a"))]
    #[case::from_function_fields(json!({"id": "call_1", "function": {"provider_specific_fields": {"thought_signature": "b"}}}), Some("b"))]
    #[case::from_id(json!({"id": "call_1__thought__c", "function": {}}), Some("c"))]
    #[case::none(json!({"id": "call_1", "function": {}}), None)]
    fn thought_signatures_come_from_where_python_looks(
        #[case] tool: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(thought_signature_from_tool(&tool), expected);
    }

    #[rstest]
    fn a_signed_call_id_is_stripped_and_its_signature_replayed() {
        let conversation = json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "calling", "provider_specific_fields": {"thought_signatures": ["s1"]}, "tool_calls": [
                {"id": "call_1__thought__s1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1__thought__s1", "content": "ok"}
        ]);
        let contents = convert_messages(&messages(conversation), "gemini-3-flash-preview").unwrap();
        assert_eq!(
            contents[1]["parts"],
            json!([
                {"text": "calling"},
                {"function_call": {"name": "f", "args": {}, "id": "call_1"}, "thoughtSignature": "s1"}
            ])
        );
        assert_eq!(contents[2]["parts"][0]["function_response"]["id"], "call_1");
    }

    #[rstest]
    #[case::no_matching_call(json!([{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "x", "content": "r"}]))]
    #[case::unknown_role(json!([{"role": "narrator", "content": "q"}]))]
    fn refuses_what_python_refuses(#[case] input: Value) {
        assert!(matches!(
            convert_messages(&messages(input), "gemini-2.5-flash"),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::object(r#"{"a": 1}"#, json!({"a": 1}))]
    #[case::list(r#"[1, 2]"#, json!({"content": "[1, 2]"}))]
    #[case::part_reference(r##"{"x": {"$ref": "#/p"}}"##, json!({"content": r##"{"x": {"$ref": "#/p"}}"##}))]
    #[case::reference_under_nested_list(r#"{"x": [[{"$ref": "p"}]]}"#, json!({"x": [[{"$ref": "p"}]]}))]
    #[case::text("plain", json!({"content": "plain"}))]
    fn tool_result_responses_parse_like_python(#[case] content: &str, #[case] expected: Value) {
        let conversation = json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": content}
        ]);
        let contents = convert_messages(&messages(conversation), "gemini-2.5-flash").unwrap();
        assert_eq!(
            contents[2]["parts"][0]["function_response"]["response"],
            expected
        );
    }

    #[rstest]
    fn a_tool_result_image_nests_inside_the_function_response() {
        let conversation = json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": [
                {"type": "text", "text": "here"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
            ]}
        ]);
        let contents = convert_messages(&messages(conversation), "gemini-2.5-flash").unwrap();
        assert_eq!(
            contents[2]["parts"],
            json!([{"function_response": {"name": "shot", "response": {"content": "here"},
                "parts": [{"inline_data": {"data": "QUJD", "mime_type": "image/png"}}]}}])
        );
    }

    #[rstest]
    #[case::gcs_extension("gs://b/x.png", None, Ok(json!({"file_data": {"mime_type": "image/png", "file_uri": "gs://b/x.png"}})))]
    #[case::gcs_format("gs://b/x", Some("image/JPG"), Ok(json!({"file_data": {"mime_type": "image/jpeg", "file_uri": "gs://b/x"}})))]
    #[case::gcs_rejected_type("gs://b/x.gif", None, Err(()))]
    #[case::https_known_type("https://h/a.pdf?sig=1", None, Ok(json!({"file_data": {"mime_type": "application/pdf", "file_uri": "https://h/a.pdf?sig=1"}})))]
    #[case::files_api("https://generativelanguage.googleapis.com/v1beta/files/abc", None, Ok(json!({"file_data": {"file_uri": "https://generativelanguage.googleapis.com/v1beta/files/abc"}})))]
    #[case::data_url("data:audio/wav;base64,UklG", None, Ok(json!({"inline_data": {"data": "UklG", "mime_type": "audio/wav"}})))]
    #[case::not_media("ftp://x", None, Err(()))]
    fn media_becomes_file_data_or_inline_data(
        #[case] url: &str,
        #[case] format: Option<&str>,
        #[case] expected: Result<Value, ()>,
    ) {
        assert_eq!(
            media_part(url, format, None, None, "gemini-2.5-flash").map_err(|_| ()),
            expected
        );
    }

    #[rstest]
    fn an_extensionless_gcs_uri_without_a_format_is_left_to_python() {
        assert_eq!(
            media_part("gs://b/object", None, None, None, "gemini-2.5-flash"),
            Err(Error::Unsupported(
                "a gs:// URI with no file extension and no format"
            ))
        );
    }

    #[rstest]
    #[case::gemini_3("gemini-3-pro-preview", true)]
    #[case::gemini_2("gemini-2.5-pro", false)]
    fn media_resolution_rides_on_the_part_only_for_gemini_3(
        #[case] model: &str,
        #[case] on_part: bool,
    ) {
        let part = image_url_part(
            &json!({"type": "image_url", "image_url": {"url": "https://h/a.png", "detail": "high"}}),
            model,
        )
        .unwrap();
        assert_eq!(part.get("media_resolution").is_some(), on_part);
    }

    #[rstest]
    fn video_metadata_is_camel_cased() {
        let part = file_part(
            &json!({"type": "file", "file": {"file_id": "gs://b/v.mp4", "video_metadata": {"fps": 2, "start_offset": "1s"}}}),
            "gemini-2.5-flash",
        )
        .unwrap();
        assert_eq!(
            part["video_metadata"],
            json!({"fps": 2, "startOffset": "1s"})
        );
    }

    #[rstest]
    #[case::strings(json!([{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]), Some(json!({"parts": [{"text": "a"}]})), 1)]
    #[case::only_system(json!([{"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]), Some(json!({"parts": [{"text": "ab"}]})), 1)]
    #[case::none(json!([{"role": "user", "content": "b"}]), None, 1)]
    fn system_messages_become_the_system_instruction(
        #[case] input: Value,
        #[case] instruction: Option<Value>,
        #[case] left: usize,
    ) {
        let (system, rest) = transform_system_message(true, messages(input));
        assert_eq!(system, instruction);
        assert_eq!(rest.len(), left);
    }

    #[rstest]
    #[case::message_level(json!({"role": "user", "content": "x", "cache_control": {"type": "ephemeral"}}), true)]
    #[case::part_level(json!({"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}), true)]
    #[case::plain(json!({"role": "user", "content": "x"}), false)]
    fn cached_messages_are_recognized(#[case] message: Value, #[case] cached: bool) {
        assert_eq!(is_cached_message(&message), cached);
    }

    #[rstest]
    #[case::snake("max_output_tokens", "maxOutputTokens")]
    #[case::single("seed", "seed")]
    fn snake_case_becomes_camel_case(#[case] snake: &str, #[case] camel: &str) {
        assert_eq!(snake_to_camel(snake), camel);
        assert_eq!(camel_to_snake(camel), snake);
    }

    fn params(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("params are an object");
        };
        map
    }

    #[rstest]
    fn the_body_sorts_params_into_their_blocks() {
        let body = transform_request_body(
            &messages(json!([{"role": "user", "content": "hi"}])),
            Some(json!({"parts": [{"text": "sys"}]})),
            "gemini-2.5-flash",
            params(json!({
                "temperature": 0.2,
                "topK": 3,
                "max_output_tokens": 10,
                "tools": [{"function_declarations": [{"name": "f"}]}],
                "tool_choice": {"functionCallingConfig": {"mode": "AUTO"}},
                "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
                "labels": {"team": "a", "n": 1},
                "service_tier": "default",
                "parallel_tool_calls": true,
                "extra_body": {"generationConfig": {"seed": 4}, "cache": {"no-cache": true}}
            })),
            GeminiSurface::Vertex,
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "system_instruction": {"parts": [{"text": "sys"}]},
                "tools": [{"function_declarations": [{"name": "f"}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
                "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
                "generationConfig": {"temperature": 0.2, "topK": 3, "max_output_tokens": 10, "seed": 4},
                "serviceTier": "standard",
                "labels": {"team": "a"}
            })
        );
    }

    #[rstest]
    #[case::ai_studio(GeminiSurface::AiStudio, false)]
    #[case::vertex(GeminiSurface::Vertex, true)]
    fn labels_reach_vertex_only(#[case] surface: GeminiSurface, #[case] sent: bool) {
        let body = transform_request_body(
            &messages(json!([{"role": "user", "content": "hi"}])),
            None,
            "gemini-2.5-flash",
            params(json!({"labels": {"team": "a"}})),
            surface,
        )
        .unwrap();
        assert_eq!(body.get("labels").is_some(), sent);
    }

    #[rstest]
    fn google_maps_with_json_output_uses_response_format() {
        let body = transform_request_body(
            &messages(json!([{"role": "user", "content": "hi"}])),
            None,
            "gemini-2.5-flash",
            params(json!({
                "tools": [{"googleMaps": {}}],
                "response_mime_type": "application/json",
                "response_json_schema": {"type": "object"}
            })),
            GeminiSurface::AiStudio,
        )
        .unwrap();
        assert_eq!(
            body["generationConfig"],
            json!({"responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": {"type": "object"}}}})
        );
    }

    #[rstest]
    fn gemini_2_takes_the_highest_image_detail_globally() {
        let body = transform_request_body(
            &messages(json!([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://h/a.png", "detail": "low"}},
                {"type": "image_url", "image_url": {"url": "https://h/b.png", "detail": "high"}}
            ]}])),
            None,
            "gemini-2.5-flash",
            Map::new(),
            GeminiSurface::Vertex,
        )
        .unwrap();
        assert_eq!(
            body["generationConfig"],
            json!({"mediaResolution": "MEDIA_RESOLUTION_HIGH"})
        );
    }

    fn chat_messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::ai_studio_fetches_every_url(GeminiSurface::AiStudio, vec!["https://h/a.png", "http://h/b", "https://h/c", "https://h/d"])]
    #[case::vertex_fetches_http_and_unknown_types(GeminiSurface::Vertex, vec!["http://h/b", "https://h/c"])]
    fn remote_media_follows_the_surface(
        #[case] surface: GeminiSurface,
        #[case] expected: Vec<&str>,
    ) {
        let list = chat_messages(json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://h/a.png"}},
            {"type": "image_url", "image_url": "http://h/b"},
            {"type": "file", "file": {"file_id": "https://h/c"}},
            {"type": "file", "file": {"file_id": "https://h/d", "format": "application/pdf"}},
            {"type": "file", "file": {"file_id": "https://generativelanguage.googleapis.com/v1beta/files/x"}},
            {"type": "image_url", "image_url": {"url": "gs://b/x.png"}}
        ]}]));
        assert_eq!(remote_media(&list, surface), expected);
    }
}
