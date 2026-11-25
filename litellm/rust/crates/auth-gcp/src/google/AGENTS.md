Code here is what Python's google-auth does and has to stay usable by any Google service, not only Vertex AI. It must not import from `crate::vertex`

Code that reads litellm settings, such as `vertex_credentials`, `vertex_project` or litellm's own `aws_*` keys, belongs in `src/vertex/` instead. The one exception is `ExplicitAwsKeys` in `credentials.rs`, which `ExternalAccountInfo` has to embed so a type error in those keys fails the credentials file the way Python does

`legacy/` holds hand-written logins for paths Google's Rust SDK lacks. Each file says which path it covers and when to delete it
