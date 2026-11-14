use std::{fmt, sync::Arc};

use futures_util::future::BoxFuture;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderOptions {
    /// Keys rendered as `null`, added the way a Python dict merge adds a missing key: last.
    pub null_keys: Vec<String>,
}

pub type Rendered = Result<Vec<Vec<u8>>, String>;

/// The Python side that owns the payloads a [`PayloadRef`] points at.
pub trait PayloadHost: Send + Sync + 'static {
    fn render(&self, handles: Vec<u64>, options: RenderOptions) -> BoxFuture<'static, Rendered>;

    fn release(&self, handle: u64);
}

/// A handle to a standard logging payload that stays in Python. A callback that reads
/// `Field::StandardLoggingPayloadRef` gets one of these instead of a JSON copy, so the payload
/// only crosses the boundary if the callback renders it. [`PayloadRef::render`] asks the host to
/// serialize it to JSON bytes; [`render_all`] does several from the same host in one call, which
/// s3_v2 uses to batch. Dropping the ref releases the host's hold on the payload.
pub struct PayloadRef {
    handle: u64,
    host: Arc<dyn PayloadHost>,
}

impl PayloadRef {
    pub fn new(handle: u64, host: Arc<dyn PayloadHost>) -> Self {
        Self { handle, host }
    }

    pub const fn handle(&self) -> u64 {
        self.handle
    }

    pub async fn render(&self, options: RenderOptions) -> Result<Vec<u8>, String> {
        let mut rendered = self.host.render(vec![self.handle], options).await?;
        match (rendered.pop(), rendered.is_empty()) {
            (Some(bytes), true) => Ok(bytes),
            _ => Err("the host rendered a different number of payloads".to_owned()),
        }
    }

    pub fn release(self) {
        drop(self);
    }
}

pub async fn render_all(payloads: &[&PayloadRef], options: RenderOptions) -> Rendered {
    let Some(first) = payloads.first() else {
        return Ok(Vec::new());
    };
    if payloads
        .iter()
        .any(|payload| !Arc::ptr_eq(&payload.host, &first.host))
    {
        return Err("payloads from different hosts cannot be rendered together".to_owned());
    }
    let rendered = first
        .host
        .render(
            payloads.iter().map(|payload| payload.handle).collect(),
            options,
        )
        .await?;
    if rendered.len() != payloads.len() {
        return Err("the host rendered a different number of payloads".to_owned());
    }
    Ok(rendered)
}

impl Drop for PayloadRef {
    fn drop(&mut self) {
        self.host.release(self.handle);
    }
}

impl fmt::Debug for PayloadRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PayloadRef")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl PartialEq for PayloadRef {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && Arc::ptr_eq(&self.host, &other.host)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::FutureExt;
    use rstest::rstest;

    use super::*;

    #[derive(Default)]
    struct Recorded {
        released: Mutex<Vec<u64>>,
        rendered: Mutex<Vec<(Vec<u64>, RenderOptions)>>,
    }

    impl PayloadHost for Recorded {
        fn render(
            &self,
            handles: Vec<u64>,
            options: RenderOptions,
        ) -> BoxFuture<'static, Rendered> {
            let bytes = handles
                .iter()
                .map(|handle| handle.to_string().into_bytes())
                .collect();
            self.rendered.lock().unwrap().push((handles, options));
            Box::pin(async move { Ok(bytes) })
        }

        fn release(&self, handle: u64) {
            self.released.lock().unwrap().push(handle);
        }
    }

    #[rstest]
    fn a_payload_is_released_once_when_its_last_owner_lets_go() {
        let host = Arc::new(Recorded::default());
        let shared = Arc::new(PayloadRef::new(
            7,
            Arc::clone(&host) as Arc<dyn PayloadHost>,
        ));
        let kept = Arc::clone(&shared);
        drop(shared);
        assert!(host.released.lock().unwrap().is_empty());
        drop(kept);
        assert_eq!(*host.released.lock().unwrap(), [7]);
    }

    #[rstest]
    fn release_frees_the_handle_at_once() {
        let host = Arc::new(Recorded::default());
        PayloadRef::new(3, Arc::clone(&host) as Arc<dyn PayloadHost>).release();
        assert_eq!(*host.released.lock().unwrap(), [3]);
    }

    #[rstest]
    fn payloads_of_one_host_render_in_one_call_in_order() {
        let host = Arc::new(Recorded::default());
        let first = PayloadRef::new(1, Arc::clone(&host) as Arc<dyn PayloadHost>);
        let second = PayloadRef::new(2, Arc::clone(&host) as Arc<dyn PayloadHost>);
        let options = RenderOptions {
            null_keys: vec!["response".into()],
        };
        let rendered = render_all(&[&first, &second], options.clone())
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_eq!(rendered, [b"1".to_vec(), b"2".to_vec()]);
        assert_eq!(*host.rendered.lock().unwrap(), [(vec![1, 2], options)]);
        assert_eq!(
            first
                .render(RenderOptions::default())
                .now_or_never()
                .unwrap(),
            Ok(b"1".to_vec())
        );
    }
}
