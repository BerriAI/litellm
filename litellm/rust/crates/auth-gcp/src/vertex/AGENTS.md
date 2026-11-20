Code here is litellm's Python `VertexBase.load_auth`, shared by every Vertex AI route: resolving `vertex_credentials`, `vertex_project` and `vertex_location`, litellm's explicit `aws_*` login, project resolution and the key-file fallback. It logs in through `crate::google` and adds nothing google-auth itself would do

Choosing where a call goes (region, URL) is not login and belongs in `litellm-llms`'s `vertex_ai` module
