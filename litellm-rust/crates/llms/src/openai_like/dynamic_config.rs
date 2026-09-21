use crate::base_llm::translation::TranslationError;
use crate::openai_like::json_loader::SimpleProviderConfig;

pub enum JsonProviderBase {
    OpenAiGpt(crate::openai::chat::gpt_transformation::OpenAiGptConfig),
    OpenAiLikeChat(crate::openai_like::chat::transformation::OpenAiLikeChatConfig),
}

pub struct JsonProviderConfig {
    pub provider: SimpleProviderConfig,
    pub base: JsonProviderBase,
}

pub struct JsonProviderResponsesConfig {
    pub provider: SimpleProviderConfig,
    pub base: crate::openai_like::responses::transformation::OpenAiLikeResponsesConfig,
}

pub fn create_config_class(
    _provider: SimpleProviderConfig,
) -> Result<JsonProviderConfig, TranslationError> {
    todo!()
}

pub fn create_responses_config_class(
    _provider: SimpleProviderConfig,
) -> Result<JsonProviderResponsesConfig, TranslationError> {
    todo!()
}

mod chat {
    use super::JsonProviderConfig;

    crate::base_llm::chat::transformation::scaffold_chat_translation_types!();
    crate::base_llm::chat::transformation::scaffold_chat_translation_config!(@existing JsonProviderConfig);
}

mod responses {
    use super::JsonProviderResponsesConfig;

    crate::base_llm::responses::transformation::scaffold_responses_translation_types!();
    crate::base_llm::responses::transformation::scaffold_responses_translation_config!(@existing JsonProviderResponsesConfig);
}
