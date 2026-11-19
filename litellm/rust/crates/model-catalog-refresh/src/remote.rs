/// Whether a host looks for a cost map beyond the bundled one
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Remote {
    /// The host serves the bundled document and never fetches
    Off,
    At(String),
}

impl Remote {
    /// The local override and offline mode each turn the remote document off
    pub fn new(url: String, local_only: bool, offline: bool) -> Self {
        match local_only || offline {
            true => Self::Off,
            false => Self::At(url),
        }
    }
}
