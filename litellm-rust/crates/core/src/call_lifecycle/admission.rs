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
