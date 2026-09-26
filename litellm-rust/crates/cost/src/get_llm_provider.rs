use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::LazyLock;

use serde_json::Value;

use crate::fallback_generalizations::FallbackGeneralizations;
use crate::provider::LlmProviders;

const OPENAI_IMAGE_GENERATION_MODELS: [&str; 2] = ["dall-e-2", "dall-e-3"];
const OPENAI_VIDEO_GENERATION_MODELS: [&str; 1] = ["sora-2"];
const COHERE_EMBEDDING_MODELS: [&str; 7] = [
    "embed-v4.0",
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0",
];
const REPLICATE_MODEL_NAME_WITH_ID_LENGTH: usize = 64;
const PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO: [&str; 2] = ["github_copilot", "chatgpt"];
const MODEL_PREFIX_PROVIDERS: [(&str, &str); 11] = [
    ("bytez/", "bytez"),
    ("gdc/", "gdc"),
    ("lemonade/", "lemonade"),
    ("heroku/", "heroku"),
    ("cometapi/", "cometapi"),
    ("oci/", "oci"),
    ("compactifai/", "compactifai"),
    ("ovhcloud/", "ovhcloud"),
    ("clarifai/", "clarifai"),
    ("amazon_nova", "amazon_nova"),
    ("sap/", "sap"),
];

static JSON_PROVIDERS: LazyLock<HashSet<String>> = LazyLock::new(|| {
    serde_json::from_str::<serde_json::Map<String, Value>>(include_str!(
        "../../../../litellm/llms/openai_like/providers.json"
    ))
    .expect("openai-like provider registry parses")
    .into_iter()
    .map(|(slug, _)| slug)
    .collect()
});

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelList {
    OpenAiChatCompletion,
    OpenAiTextCompletion,
    Anthropic,
    Cohere,
    CohereChat,
    MistralChat,
    Openrouter,
    VertexText,
    VertexCodeText,
    VertexLanguage,
    VertexVision,
    VertexChat,
    VertexCodeChat,
    VertexEmbedding,
    VertexLlama3,
    VertexMistral,
    VertexAi21,
    VertexImage,
    VertexVideo,
    Ai21,
    NlpCloud,
    AlephAlpha,
    Bedrock,
    BedrockConverse,
    Watsonx,
    Empower,
    GradientAi,
}

fn is_openai_finetune_model(key: &str) -> bool {
    key.starts_with("ft:") && key.matches(':').count() <= 1
}

fn vertex_member(key: &str) -> String {
    key.replace("vertex_ai/", "")
}

fn model_list_membership(key: &str, info: &Value) -> Option<(ModelList, String)> {
    let provider = info.get("litellm_provider").and_then(Value::as_str)?;
    let mode = info.get("mode").and_then(Value::as_str);
    let list = match provider {
        "openai" if !is_openai_finetune_model(key) => ModelList::OpenAiChatCompletion,
        "text-completion-openai" => ModelList::OpenAiTextCompletion,
        "cohere" => ModelList::Cohere,
        "cohere_chat" => ModelList::CohereChat,
        "mistral" => ModelList::MistralChat,
        "anthropic" => ModelList::Anthropic,
        "empower" => ModelList::Empower,
        "openrouter" => ModelList::Openrouter,
        "vertex_ai-text-models" => ModelList::VertexText,
        "vertex_ai-code-text-models" => ModelList::VertexCodeText,
        "vertex_ai-language-models" => ModelList::VertexLanguage,
        "vertex_ai-vision-models" => ModelList::VertexVision,
        "vertex_ai-chat-models" => ModelList::VertexChat,
        "vertex_ai-code-chat-models" => ModelList::VertexCodeChat,
        "vertex_ai-embedding-models" => ModelList::VertexEmbedding,
        "vertex_ai-llama_models" => return Some((ModelList::VertexLlama3, vertex_member(key))),
        "vertex_ai-mistral_models" => return Some((ModelList::VertexMistral, vertex_member(key))),
        "vertex_ai-ai21_models" => return Some((ModelList::VertexAi21, vertex_member(key))),
        "vertex_ai-image-models" => return Some((ModelList::VertexImage, vertex_member(key))),
        "vertex_ai-video-models" => return Some((ModelList::VertexVideo, vertex_member(key))),
        "ai21" => ModelList::Ai21,
        "nlp_cloud" => ModelList::NlpCloud,
        "aleph_alpha" => ModelList::AlephAlpha,
        "bedrock" if mode == Some("guardrail") => return None,
        "bedrock" => ModelList::Bedrock,
        "bedrock_converse" => ModelList::BedrockConverse,
        "watsonx" => ModelList::Watsonx,
        "gradient_ai" => ModelList::GradientAi,
        _ => return None,
    };
    Some((list, key.to_owned()))
}

#[derive(Clone, Debug, Default)]
pub struct ProviderModelSets {
    members: HashMap<ModelList, HashSet<String>>,
}

impl ProviderModelSets {
    pub fn from_model_cost(model_cost: &HashMap<String, Value>) -> Self {
        let members = model_cost
            .iter()
            .filter_map(|(key, info)| model_list_membership(key, info))
            .fold(
                HashMap::<ModelList, HashSet<String>>::new(),
                |mut members, (list, name)| {
                    members.entry(list).or_default().insert(name);
                    members
                },
            );
        Self { members }
    }

    pub fn contains(&self, list: ModelList, model: &str) -> bool {
        self.members
            .get(&list)
            .is_some_and(|models| models.contains(model))
    }

    fn contains_any(&self, lists: &[ModelList], model: &str) -> bool {
        lists.iter().any(|list| self.contains(*list, model))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlmProvider {
    pub model: String,
    pub custom_llm_provider: String,
}

pub fn declared_authenticating_provider(model: &str) -> Option<&str> {
    model
        .split_once('/')
        .map(|(declared, _)| declared)
        .filter(|declared| PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO.contains(declared))
}

fn is_anthropic_text_model(model: &str) -> bool {
    matches!(model, "claude-2" | "claude-instant-1")
}

fn resolved(model: &str, provider: &str) -> Option<LlmProvider> {
    Some(LlmProvider {
        model: model.to_owned(),
        custom_llm_provider: provider.to_owned(),
    })
}

fn bare_model_provider<'a>(model: &str, sets: &ProviderModelSets) -> Option<&'a str> {
    use ModelList::*;
    let replicate_id = model
        .split(':')
        .nth(1)
        .is_some_and(|id| id.len() == REPLICATE_MODEL_NAME_WITH_ID_LENGTH);
    if sets.contains(OpenAiChatCompletion, model)
        || model.contains("ft:gpt-3.5-turbo")
        || model.contains("ft:gpt-4")
        || OPENAI_IMAGE_GENERATION_MODELS.contains(&model)
        || model.starts_with("gpt-image")
        || OPENAI_VIDEO_GENERATION_MODELS.contains(&model)
    {
        Some("openai")
    } else if sets.contains(OpenAiTextCompletion, model) {
        Some("text-completion-openai")
    } else if sets.contains(Anthropic, model) {
        Some(if is_anthropic_text_model(model) {
            "anthropic_text"
        } else {
            "anthropic"
        })
    } else if sets.contains(Cohere, model) || COHERE_EMBEDDING_MODELS.contains(&model) {
        Some("cohere")
    } else if sets.contains(CohereChat, model) {
        Some("cohere_chat")
    } else if model.contains(':') && model.len() > REPLICATE_MODEL_NAME_WITH_ID_LENGTH {
        replicate_id.then_some("replicate")
    } else if sets.contains(Openrouter, model) {
        Some("openrouter")
    } else if sets.contains_any(
        &[
            VertexChat,
            VertexCodeChat,
            VertexText,
            VertexCodeText,
            VertexLanguage,
            VertexEmbedding,
            VertexVision,
            VertexImage,
            VertexVideo,
        ],
        model,
    ) {
        Some("vertex_ai")
    } else if sets.contains(Ai21, model) {
        Some("ai21_chat")
    } else if sets.contains(AlephAlpha, model) {
        Some("aleph_alpha")
    } else if sets.contains(NlpCloud, model) {
        Some("nlp_cloud")
    } else if sets.contains_any(&[Bedrock, BedrockConverse], model) {
        Some("bedrock")
    } else if sets.contains(Watsonx, model) {
        Some("watsonx")
    } else if sets.contains(Empower, model) {
        Some("empower")
    } else if sets.contains(GradientAi, model) {
        Some("gradient_ai")
    } else if model == "*" {
        Some("openai")
    } else {
        MODEL_PREFIX_PROVIDERS
            .iter()
            .find(|(prefix, _)| model.starts_with(prefix))
            .map(|(_, provider)| *provider)
    }
}

pub fn get_llm_provider(
    model: &str,
    sets: &ProviderModelSets,
    generalizations: &FallbackGeneralizations,
) -> Option<LlmProvider> {
    if let Some((prefix, rest)) = model.split_once('/') {
        if prefix == "azure"
            && (sets.contains(ModelList::CohereChat, rest)
                || sets.contains(ModelList::MistralChat, &format!("mistral/{rest}")))
        {
            return resolved(model, "openai");
        }
        if prefix == "cohere" && sets.contains(ModelList::CohereChat, rest) {
            return resolved(rest, "cohere_chat");
        }
        if prefix == "anthropic" && is_anthropic_text_model(rest) {
            return resolved(rest, "anthropic_text");
        }
        if JSON_PROVIDERS.contains(prefix) {
            return resolved(rest, prefix);
        }
        if LlmProviders::from_str(prefix).is_ok() {
            let provider = if prefix == "ai21" {
                "ai21_chat"
            } else {
                prefix
            };
            return resolved(rest, provider);
        }
    }
    bare_model_provider(model, sets)
        .or_else(|| generalizations.match_routing(model))
        .and_then(|provider| resolved(model, provider))
}
