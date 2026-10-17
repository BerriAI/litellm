//! Media a chat message points at. Python downloads an image or file URL a provider cannot
//! take by reference while it builds the request; here the host downloads them first, from
//! the list a config names, and hands the messages back with each one inlined as a data URL.

use std::collections::HashMap;

use litellm_llms_types::formats::chat_completions::{ChatMessage, ChatMessageContent};
use serde_json::Value;

/// Python's `MAX_IMAGE_URL_DOWNLOAD_SIZE_MB`.
pub const MAX_MEDIA_DOWNLOAD_BYTES: u64 = 50 * 1024 * 1024;

/// A part's `image_url`, given as a string or as `{"url": ...}`.
pub fn image_url(part: &Value) -> Option<&str> {
    let image_url = part.get("image_url")?;
    image_url
        .as_str()
        .or_else(|| image_url.get("url").and_then(Value::as_str))
}

/// `data:<meta>;base64,<data>` split into its media type and payload, the way Python's
/// `split("data:")[1].split(";base64,")` reads it.
pub fn data_url(url: &str) -> Option<(&str, &str)> {
    let (_, rest) = url.split_once("data:")?;
    let (media_type, data) = rest.split_once(";base64,")?;
    (!data.contains(";base64,")).then_some((media_type, data))
}

fn parts(message: &ChatMessage) -> &[Value] {
    match &message.content {
        Some(ChatMessageContent::Parts(parts)) => parts,
        _ => &[],
    }
}

/// The URLs of `image_url` parts (and, with `files`, of `file.file_id` / `file.file_data`)
/// in the messages `roles` names that `remote` says must be downloaded, in order, each once.
pub fn remote_urls(
    messages: &[ChatMessage],
    roles: &[&str],
    files: bool,
    remote: impl Fn(&str) -> bool,
) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    let mut add = |url: &str| {
        if remote(url) && !urls.iter().any(|seen| seen == url) {
            urls.push(url.to_owned());
        }
    };
    for message in messages
        .iter()
        .filter(|message| roles.contains(&message.role.as_str()))
    {
        for part in parts(message) {
            match part.get("type").and_then(Value::as_str) {
                Some("image_url") => {
                    if let Some(url) = image_url(part) {
                        add(url);
                    }
                }
                Some("file") if files => {
                    let file = part.get("file");
                    let source = ["file_id", "file_data"].into_iter().find_map(|key| {
                        file.and_then(|file| file.get(key)).and_then(Value::as_str)
                    });
                    if let Some(url) = source {
                        add(url);
                    }
                }
                _ => {}
            }
        }
    }
    urls
}

/// Every reference to a downloaded URL replaced by its data URL.
pub fn inline_media(messages: &mut [ChatMessage], downloaded: &HashMap<String, String>) {
    if downloaded.is_empty() {
        return;
    }
    for message in messages {
        let Some(ChatMessageContent::Parts(parts)) = &mut message.content else {
            continue;
        };
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("image_url") => match part.get_mut("image_url") {
                    Some(Value::String(url)) => {
                        if let Some(data) = downloaded.get(url.as_str()) {
                            *url = data.clone();
                        }
                    }
                    Some(Value::Object(image)) => {
                        if let Some(Value::String(url)) = image.get_mut("url")
                            && let Some(data) = downloaded.get(url.as_str())
                        {
                            *url = data.clone();
                        }
                    }
                    _ => {}
                },
                Some("file") => {
                    if let Some(Value::Object(file)) = part.get_mut("file") {
                        for key in ["file_id", "file_data"] {
                            if let Some(Value::String(url)) = file.get_mut(key)
                                && let Some(data) = downloaded.get(url.as_str())
                            {
                                *url = data.clone();
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// The media type a download is inlined with: the response's own, or one guessed from the
/// URL's extension the way Python's `infer_content_type_from_url_and_content` falls back.
pub fn downloaded_media_type(url: &str, content_type: &str) -> String {
    let declared = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !declared.is_empty() && declared != "application/octet-stream" && declared.contains('/') {
        return declared;
    }
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .to_ascii_lowercase();
    let extension = path.rsplit('.').next().unwrap_or_default();
    match extension {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "html" | "htm" => "text/html",
        "md" => "text/markdown",
        "mp4" => "video/mp4",
        _ => "image/jpeg",
    }
    .to_owned()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    fn remote_urls_are_listed_once_and_inlined() {
        let mut list = messages(json!([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": "http://a/x.png"},
                {"type": "image_url", "image_url": {"url": "http://a/x.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "file", "file": {"file_id": "https://b/doc.pdf"}}
            ]},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": "http://c"}]}
        ]));
        let urls = remote_urls(&list, &["user", "tool"], true, |url| {
            url.starts_with("http")
        });
        assert_eq!(urls, vec!["http://a/x.png", "https://b/doc.pdf"]);
        let downloaded = HashMap::from([(
            "http://a/x.png".to_owned(),
            "data:image/png;base64,Zm8=".to_owned(),
        )]);
        inline_media(&mut list, &downloaded);
        let content = serde_json::to_value(&list[0].content).unwrap();
        assert_eq!(content[0]["image_url"], "data:image/png;base64,Zm8=");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,Zm8=");
        assert_eq!(content[3]["file"]["file_id"], "https://b/doc.pdf");
    }

    #[rstest]
    #[case("data:image/png;base64,AAAA", Some(("image/png", "AAAA")))]
    #[case("https://x", None)]
    fn data_urls_split(#[case] url: &str, #[case] expected: Option<(&str, &str)>) {
        assert_eq!(data_url(url), expected);
    }

    #[rstest]
    #[case("https://x/a.PNG?x=1", "", "image/png")]
    #[case("https://x/a", "image/webp; charset=x", "image/webp")]
    #[case("https://x/a.pdf", "application/octet-stream", "application/pdf")]
    fn media_types_fall_back_to_the_extension(
        #[case] url: &str,
        #[case] content_type: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(downloaded_media_type(url, content_type), expected);
    }
}
