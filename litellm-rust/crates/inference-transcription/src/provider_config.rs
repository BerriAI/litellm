use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_inference::provider::{ResolvedProvider, resolve_llm_provider};
use litellm_llms::{
    base_llm::audio_transcription::transformation::BaseAudioTranscriptionConfig,
    bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG,
};

use crate::Error;

fn provider_config(provider: LlmProviders) -> Option<&'static dyn BaseAudioTranscriptionConfig> {
    match provider {
        LlmProviders::Bedrock => Some(&BEDROCK_AUDIO_TRANSCRIPTION_CONFIG),
        _ => None,
    }
}

pub(crate) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<
    (
        ResolvedProvider<'a>,
        &'static dyn BaseAudioTranscriptionConfig,
    ),
    Error,
> {
    let provider = resolve_llm_provider(model, custom_llm_provider, "audio transcription")?;
    let config = provider_config(provider.provider)
        .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?;
    Ok((provider, config))
}
