use std::convert::Infallible;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
#[strum(serialize_all = "snake_case")]
pub enum UnimplementedRoute {
    Messages,
    ChatCompletions,
    Transcription,
    Embeddings,
    Rerank,
    ImageGeneration,
    ImageEdit,
    Speech,
    Moderation,
    Responses,
}

pub fn admit_unimplemented(route: UnimplementedRoute) -> Result<Infallible, UnimplementedRoute> {
    Err(route)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub enum AdmissionDecline {
    #[strum(to_string = "provider is not supported by this native route")]
    Provider,
    #[strum(to_string = "required host operations are not supported")]
    HostOperations,
    #[strum(to_string = "request contains values that cannot be inspected without Python effects")]
    Uninspectable,
    #[strum(to_string = "{0}")]
    Feature(&'static str),
}

pub enum Inspection<T> {
    Inspectable(T),
    Uninspectable,
}
