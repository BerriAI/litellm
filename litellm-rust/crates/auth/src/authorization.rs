#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyAuthorizationInput<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub body: &'a [u8],
}
