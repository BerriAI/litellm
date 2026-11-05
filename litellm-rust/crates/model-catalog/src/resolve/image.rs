/// What an image call asks for. `resolve_image` builds main's eight key spellings from it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageCandidates {
    pub model: String,
    pub provider: Option<String>,
    /// Width and height, `1024 x 1024` when the request names none, as on main
    pub size: Option<(u32, u32)>,
    pub quality: Option<String>,
}

/// A size and quality keyed row, `{quality}/{W}-x-{H}/{model}`, parsed once per index build for fal's nearest match
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageKey {
    pub quality: Option<String>,
    pub width: u32,
    pub height: u32,
    pub model: String,
}
