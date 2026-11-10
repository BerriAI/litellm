use serde_json::{Map, Value};

use crate::formats::messages::{CacheControl, CitationsConfig, ContentSource};

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ImageUrl {
        image_url: ChatMediaUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    VideoUrl {
        video_url: ChatMediaUrl,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputAudio {
        input_audio: ChatInputAudio,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    File {
        file: Box<ChatFile>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Document {
        source: Box<ContentSource>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Refusal {
        refusal: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct PromptCacheBreakpoint {
    pub mode: PromptCacheMode,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheMode {
    Explicit,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ChatMediaUrl {
    Url(String),
    Parameters(Box<ChatMediaUrlParameters>),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatMediaUrlParameters {
    pub url: String,
    pub detail: Option<String>,
    pub format: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatInputAudio {
    pub data: String,
    pub format: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatFile {
    pub file_data: Option<String>,
    pub file_id: Option<String>,
    pub filename: Option<String>,
    pub format: Option<String>,
    pub detail: Option<String>,
    pub video_metadata: Option<ChatVideoMetadata>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatVideoMetadata {
    pub fps: Option<serde_json::Number>,
    pub start_offset: Option<String>,
    pub end_offset: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatLogprobs {
    pub content: Option<Vec<ChatTokenLogprob>>,
    pub refusal: Option<Vec<ChatTokenLogprob>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatTokenLogprob {
    pub token: String,
    pub logprob: serde_json::Number,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Option<Vec<ChatTopLogprob>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatTopLogprob {
    pub token: String,
    pub logprob: serde_json::Number,
    pub bytes: Option<Vec<u8>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use crate::formats::{
        chat_completions::{ChatContentPart, ChatLogprobs, ChatMediaUrl, PromptCacheMode},
        messages::ContentSource,
    };

    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::parts(json!([
        {"type":"text","text":"hello","cache_control":{"type":"ephemeral"},"prompt_cache_breakpoint":{"mode":"explicit"}},
        {"type":"image_url","image_url":{"url":"https://example.test/image","detail":"high","format":"image/png"}},
        {"type":"video_url","video_url":"https://example.test/video"},
        {"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}},
        {"type":"file","file":{"file_id":"file_1","file_data":"JVBE","filename":"a.mp4","format":"video/mp4","detail":"low","video_metadata":{"fps":1,"start_offset":"1s","end_offset":"2s"}}},
        {"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"title":"T","context":"C","citations":{"enabled":true}},
        {"type":"refusal","refusal":"refused"}
    ]))]
    fn content_parts_round_trip(#[case] wire: Value) {
        let parts: Vec<ChatContentPart> = serde_json::from_value(wire.clone()).unwrap();
        let [
            ChatContentPart::Text {
                text,
                cache_control,
                prompt_cache_breakpoint: Some(breakpoint),
                extra: text_extra,
            },
            ChatContentPart::ImageUrl {
                image_url: ChatMediaUrl::Parameters(image),
                prompt_cache_breakpoint: None,
                ..
            },
            ChatContentPart::VideoUrl {
                video_url: ChatMediaUrl::Url(video),
                ..
            },
            ChatContentPart::InputAudio { input_audio, .. },
            ChatContentPart::File { file, .. },
            ChatContentPart::Document {
                source,
                title: Some(title),
                context: Some(context),
                citations: Some(citations),
                ..
            },
            ChatContentPart::Refusal { refusal, .. },
        ] = parts.as_slice()
        else {
            panic!("expected typed content parts");
        };
        assert_eq!(text, "hello");
        assert_eq!(
            cache_control.as_ref().unwrap().cache_type.as_deref(),
            Some("ephemeral")
        );
        assert_eq!(breakpoint.mode, PromptCacheMode::Explicit);
        assert!(breakpoint.extra.is_empty());
        assert!(text_extra.is_empty());
        assert_eq!(image.url, "https://example.test/image");
        assert_eq!(image.detail.as_deref(), Some("high"));
        assert_eq!(image.format.as_deref(), Some("image/png"));
        assert!(image.extra.is_empty());
        assert_eq!(video, "https://example.test/video");
        assert_eq!(input_audio.data, "AA==");
        assert_eq!(input_audio.format, "wav");
        assert_eq!(file.file_id.as_deref(), Some("file_1"));
        assert_eq!(file.file_data.as_deref(), Some("JVBE"));
        assert_eq!(file.filename.as_deref(), Some("a.mp4"));
        assert_eq!(file.format.as_deref(), Some("video/mp4"));
        assert_eq!(file.detail.as_deref(), Some("low"));
        assert!(file.extra.is_empty());
        let metadata = file.video_metadata.as_ref().unwrap();
        assert_eq!(metadata.fps, Some(1.into()));
        assert_eq!(metadata.start_offset.as_deref(), Some("1s"));
        assert_eq!(metadata.end_offset.as_deref(), Some("2s"));
        assert!(metadata.extra.is_empty());
        let ContentSource::Text {
            data, media_type, ..
        } = source.as_ref()
        else {
            panic!("expected document text source");
        };
        assert_eq!(data, "doc");
        assert_eq!(media_type, "text/plain");
        assert_eq!(title, "T");
        assert_eq!(context, "C");
        assert_eq!(citations.enabled, Some(true));
        assert_eq!(refusal, "refused");
        assert_eq!(serde_json::to_value(parts).unwrap(), wire);
    }

    #[rstest]
    fn logprobs_round_trip_with_tokens_and_alternatives() {
        let wire = json!({
            "content":[{"token":"hi","logprob":-1,"bytes":[104,105],"top_logprobs":[{"token":"hey","logprob":-2.5,"bytes":[104]}]}],
            "refusal":[{"token":"refused","logprob":-3}]
        });
        let logprobs: ChatLogprobs = serde_json::from_value(wire.clone()).unwrap();
        let token = &logprobs.content.as_ref().unwrap()[0];
        assert_eq!(token.token, "hi");
        assert_eq!(token.logprob, (-1).into());
        assert_eq!(token.bytes.as_deref(), Some([104, 105].as_slice()));
        let alternative = &token.top_logprobs.as_ref().unwrap()[0];
        assert_eq!(alternative.token, "hey");
        assert_eq!(alternative.bytes.as_deref(), Some([104].as_slice()));
        assert_eq!(logprobs.refusal.as_ref().unwrap()[0].token, "refused");
        assert!(logprobs.refusal.as_ref().unwrap()[0].top_logprobs.is_none());
        assert_eq!(serde_json::to_value(logprobs).unwrap(), wire);
    }

    #[rstest]
    #[case::text(json!({"type":"text","text":false}))]
    #[case::missing_image(json!({"type":"image_url"}))]
    #[case::audio_shape(json!({"type":"input_audio","input_audio":{"data":7,"format":"wav"}}))]
    #[case::audio_missing_format(json!({"type":"input_audio","input_audio":{"data":"AA=="}}))]
    #[case::image_missing_url(json!({"type":"image_url","image_url":{"detail":"high"}}))]
    #[case::file_metadata(json!({"type":"file","file":{"video_metadata":{"fps":"fast"}}}))]
    #[case::document_source(json!({"type":"document","source":{"type":"url","url":false}}))]
    #[case::refusal_missing_text(json!({"type":"refusal"}))]
    #[case::file_missing_file(json!({"type":"file"}))]
    #[case::document_missing_source(json!({"type":"document"}))]
    #[case::video_missing_url(json!({"type":"video_url"}))]
    #[case::cache_control_shape(json!({"type":"text","text":"t","cache_control":"ephemeral"}))]
    #[case::breakpoint_missing_mode(json!({"type":"text","text":"t","prompt_cache_breakpoint":{}}))]
    #[case::breakpoint_unknown_mode(json!({"type":"text","text":"t","prompt_cache_breakpoint":{"mode":"auto"}}))]
    #[case::unknown_tag(json!({"type":"future"}))]
    fn content_parts_reject_malformed_typed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<ChatContentPart>(wire).is_err());
    }

    #[rstest]
    fn partial_file_preserves_extensions_and_omits_null_optionals() {
        let wire = json!({"type":"file","file":{"file_id":null,"filename":"a.pdf","extension":[1,null]},"future":true});
        let part: ChatContentPart = serde_json::from_value(wire).unwrap();
        let ChatContentPart::File {
            file,
            prompt_cache_breakpoint: None,
            extra,
        } = &part
        else {
            panic!("expected file")
        };
        assert!(file.file_id.is_none());
        assert!(file.file_data.is_none());
        assert_eq!(file.filename.as_deref(), Some("a.pdf"));
        assert_eq!(extra["future"], json!(true));
        assert_eq!(
            serde_json::to_value(part).unwrap(),
            json!({"type":"file","file":{"filename":"a.pdf","extension":[1,null]},"future":true})
        );
    }

    #[rstest]
    #[case::missing_token(json!({"logprob":-1}))]
    #[case::missing_logprob(json!({"token":"hi"}))]
    #[case::wrong_bytes(json!({"token":"hi","logprob":-1,"bytes":[256]}))]
    fn token_logprobs_reject_malformed_fields(#[case] wire: Value) {
        assert!(
            serde_json::from_value::<crate::formats::chat_completions::ChatTokenLogprob>(wire)
                .is_err()
        );
    }
}
