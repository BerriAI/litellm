use crate::openai_like::json_loader::SimpleProviderConfig;

crate::base_llm::anthropic_messages::transformation::scaffold_messages_translation_types!();

pub struct JsonProviderAnthropicMessagesConfig {
    pub provider: SimpleProviderConfig,
}

impl JsonProviderAnthropicMessagesConfig {
    pub fn new(_provider: SimpleProviderConfig) -> Self {
        todo!()
    }
}

crate::base_llm::anthropic_messages::transformation::scaffold_messages_translation_config!(
    @existing JsonProviderAnthropicMessagesConfig
);
