use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ResponseEnvelope<T> {
    version: u32,
    surface: String,
    output: T,
}

impl<T> ResponseEnvelope<T> {
    pub fn new(surface: &str, output: T) -> Self {
        Self {
            version: 1,
            surface: surface.into(),
            output,
        }
    }

    pub fn decode(self, surface: &str) -> Option<T> {
        (self.version == 1 && self.surface == surface).then_some(self.output)
    }
}
