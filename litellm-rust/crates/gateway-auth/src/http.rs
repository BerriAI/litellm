use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use litellm_auth_types::SecretValue;

use crate::{Auth, Error};

pub async fn authenticate<E: Into<SecretValue>>(
    State(auth): State<Auth>,
    credential: E,
    mut request: Request,
    next: Next,
) -> Result<Response, Error> {
    let caller = auth.authenticate(&credential.into()).await?;
    request.extensions_mut().insert(caller);
    Ok(next.run(request).await)
}
