#[derive(Debug, rustify_derive::Endpoint)]
#[endpoint(path = "/auth/{self.mount}/login", method = "POST")]
pub struct CertLoginRequest {
    #[endpoint(skip)]
    pub mount: String,
    #[endpoint(raw)]
    body: Vec<u8>,
}

impl CertLoginRequest {
    pub fn new(name: Option<&str>) -> Self {
        let body: Vec<u8> = match name {
            Some(name) => serde_json::to_vec(&serde_json::json!({ "name": name }))
                .expect("json object serialization is infallible"),
            None => b"{}".to_vec(),
        };
        Self {
            mount: "cert".to_owned(),
            body,
        }
    }
}
