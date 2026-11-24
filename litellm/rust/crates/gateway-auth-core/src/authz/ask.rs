//! What a request asks for, as far as authz reads it. A host resolves it once, so every layer of
//! a caller is judged against the same names.

/// The model a request asks for.
#[derive(Clone, Copy, Debug)]
pub struct ModelAsk<'a> {
    /// The name as the request spells it, which a denial quotes back.
    pub name: &'a str,
    /// Every name a list may grant the model under, the request's own first, as
    /// [`candidates`](super::candidates) gives them.
    pub candidates: &'a [&'a str],
    /// The access groups of the candidates' deployments, each of which a list may name in the
    /// model's place.
    pub access_groups: &'a [String],
    /// A candidate is a name the caller's team aliases. Aliasing a name is the team's own grant,
    /// so the key's and the team's lists reach it whatever they hold.
    pub team_alias: bool,
}
