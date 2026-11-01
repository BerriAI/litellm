/// Primary API surface / task type of the model.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    AudioSpeech,
    AudioTranscription,
    Chat,
    Completion,
    Decisions,
    Embedding,
    Evaluation,
    Extraction,
    Guardrail,
    ImageEdit,
    ImageGeneration,
    Moderation,
    Ocr,
    Realtime,
    Rerank,
    Responses,
    Search,
    VectorStore,
    VideoGeneration,
}

/// Gemini audio generation API the model is served through.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum VertexAiAudioApi {
    LyriaPredict,
    LyriaInteractions,
}

/// Audio container format the model can return.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Mp3,
    Wav,
}

/// Input modality the model accepts.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum InputModality {
    Text,
    Image,
    Audio,
    Video,
}

/// Output modality the model can produce.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OutputModality {
    Text,
    Image,
    Audio,
    Video,
    Code,
}

/// API route a model can be called through.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
pub enum Endpoint {
    #[serde(rename = "/v1/chat/completions")]
    ChatCompletions,
    #[serde(rename = "/v1/completions")]
    Completions,
    #[serde(rename = "/v1/responses")]
    Responses,
    #[serde(rename = "/v1/messages")]
    Messages,
    #[serde(rename = "/v1/batch")]
    Batch,
    #[serde(rename = "/v1/embeddings")]
    Embeddings,
    #[serde(rename = "/v1/images/generations")]
    ImageGenerations,
    #[serde(rename = "/v1/images/edits")]
    ImageEdits,
    #[serde(rename = "/v1/audio/speech")]
    AudioSpeech,
    #[serde(rename = "/v1/audio/transcriptions")]
    AudioTranscriptions,
    #[serde(rename = "/v1/realtime")]
    Realtime,
    #[serde(rename = "/v1/realtime/transcription_sessions")]
    RealtimeTranscriptionSessions,
    #[serde(rename = "/v1/videos")]
    Videos,
    #[serde(rename = "/v1/ocr")]
    Ocr,
    #[serde(rename = "/v1/listen")]
    Listen,
    #[serde(rename = "/v1/systemone")]
    SystemOne,
    #[serde(rename = "/v1/decisions")]
    Decisions,
    #[serde(rename = "/v1beta/interactions")]
    Interactions,
    #[serde(rename = "/v1beta/models")]
    Models,
    #[serde(rename = "/vertex_ai/live")]
    VertexAiLive,
}

/// Reasoning effort level a model accepts or applies.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EffortLevel {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}
