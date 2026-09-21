#[derive(Debug, rustify_derive::Endpoint)]
#[endpoint(path = "/auth/{self.mount}/login", method = "POST")]
pub struct CertLoginRequest {
    #[endpoint(skip)]
    pub mount: String,
    #[endpoint(skip)]
    #[allow(dead_code)]
    pub name: Option<String>,
    #[endpoint(raw)]
    body: Vec<u8>,
}

impl CertLoginRequest {
    pub fn new(name: Option<String>) -> Self {
        let body: Vec<u8> = match name.as_deref() {
            Some(name) => serde_json::to_vec(&serde_json::json!({ "name": name })).unwrap(),
            None => b"{}".to_vec(),
        };
        Self {
            mount: "cert".to_owned(),
            name,
            body,
        }
    }
}
