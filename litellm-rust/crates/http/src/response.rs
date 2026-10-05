use http::{HeaderMap, StatusCode};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseHead {
    pub status: StatusCode,
    pub headers: HeaderMap,
}

impl ResponseHead {
    pub fn from_response(response: &reqwest::Response) -> Self {
        Self {
            status: response.status(),
            headers: response.headers().clone(),
        }
    }

    pub fn cached() -> Self {
        Self {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response<T> {
    pub head: ResponseHead,
    pub body: T,
}

impl<T> Response<T> {
    pub fn cached(body: T) -> Self {
        Self {
            head: ResponseHead::cached(),
            body,
        }
    }

    pub fn map<U>(self, transform: impl FnOnce(T) -> U) -> Response<U> {
        Response {
            head: self.head,
            body: transform(self.body),
        }
    }
}
