use crate::responses_usage::{
    CacheCreationTokenDetails, ChatUsage, text_tokens_without_nested_reasoning,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedPromptDetails {
    pub cache_hit_tokens: u64,
    pub cache_hit_audio_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_creation_token_details: Option<CacheCreationTokenDetails>,
    pub text_tokens: u64,
    pub audio_tokens: u64,
    pub image_tokens: u64,
    pub video_tokens: u64,
    pub character_count: u64,
    pub image_count: u64,
    pub video_length_seconds: f64,
    pub audio_length_seconds: f64,
    pub query_count: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ParsedCompletionDetails {
    pub audio_tokens: u64,
    pub text_tokens: u64,
    pub reasoning_tokens: u64,
    pub image_tokens: u64,
    pub video_tokens: u64,
}

pub fn parse_prompt_tokens_details(usage: &ChatUsage) -> ParsedPromptDetails {
    let Some(details) = usage.prompt_tokens_details.as_ref() else {
        return ParsedPromptDetails::default();
    };
    let cached = details.cached_tokens;
    let cached_details = details.cached_tokens_details.as_ref();
    let cached_audio = cached_details
        .and_then(|details| details.audio_tokens)
        .unwrap_or(0)
        .min(cached);
    let cached_text = cached_details
        .and_then(|details| details.text_tokens)
        .unwrap_or(0)
        .min(cached - cached_audio);
    let cached_image = cached_details
        .and_then(|details| details.image_tokens)
        .unwrap_or(0)
        .min(cached - cached_audio - cached_text);
    ParsedPromptDetails {
        cache_hit_tokens: cached,
        cache_hit_audio_tokens: cached_audio,
        cache_creation_tokens: details
            .cache_write_tokens
            .filter(|tokens| *tokens > 0)
            .or(details.cache_creation_tokens)
            .unwrap_or(0),
        cache_creation_token_details: details.cache_creation_token_details.clone(),
        text_tokens: details.text_tokens.unwrap_or(0).saturating_sub(cached_text),
        audio_tokens: details
            .audio_tokens
            .unwrap_or(0)
            .saturating_sub(cached_audio),
        image_tokens: details
            .image_tokens
            .unwrap_or(0)
            .saturating_sub(cached_image),
        video_tokens: details.video_tokens.unwrap_or(0),
        character_count: details.character_count.unwrap_or(0),
        image_count: details.image_count.unwrap_or(0),
        video_length_seconds: details.video_length_seconds.unwrap_or(0.0),
        audio_length_seconds: details.audio_length_seconds.unwrap_or(0.0),
        query_count: details.query_count.unwrap_or(0),
    }
}

pub fn get_billable_input_tokens(prompt_tokens: u64, cache_hit_tokens: u64) -> i128 {
    prompt_tokens as i128 - cache_hit_tokens as i128
}

pub fn parse_completion_tokens_details(usage: &ChatUsage) -> ParsedCompletionDetails {
    let Some(details) = usage.completion_tokens_details.as_ref() else {
        return ParsedCompletionDetails::default();
    };
    let audio = details.audio_tokens.unwrap_or(0);
    let image = details.image_tokens.unwrap_or(0);
    let video = details.video_tokens.unwrap_or(0);
    let reasoning = details.reasoning_tokens.unwrap_or(0);
    ParsedCompletionDetails {
        audio_tokens: audio,
        text_tokens: text_tokens_without_nested_reasoning(
            usage.completion_tokens,
            details.text_tokens.unwrap_or(0),
            reasoning,
            audio.saturating_add(image).saturating_add(video),
        ),
        reasoning_tokens: reasoning,
        image_tokens: image,
        video_tokens: video,
    }
}
