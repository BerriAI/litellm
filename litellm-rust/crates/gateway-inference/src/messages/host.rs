use std::{convert::Infallible, sync::Mutex};

use bytes::Bytes;
use litellm_core::messages::{
    Error,
    route::{LocalMessagesHost, Messages, MessagesCall, MessagesStreamHead},
};
use litellm_host::host::{Demand, Host};
use tokio::sync::{mpsc, oneshot};

/// Hands a streamed response to the HTTP body: the head once, then each chunk. A dropped
/// receiver means the client went away, which detaches the call.
pub(super) struct ChannelHost {
    local: LocalMessagesHost,
    head: Mutex<Option<oneshot::Sender<MessagesStreamHead>>>,
    pub(super) chunks: mpsc::Sender<Bytes>,
}

impl ChannelHost {
    pub(super) fn new(
        call: MessagesCall,
        head: oneshot::Sender<MessagesStreamHead>,
        chunks: mpsc::Sender<Bytes>,
    ) -> Self {
        Self {
            local: LocalMessagesHost::new(call),
            head: Mutex::new(Some(head)),
            chunks,
        }
    }

    fn take_head(&self) -> Option<oneshot::Sender<MessagesStreamHead>> {
        self.head
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
    }

    pub(super) fn opened(&self) -> bool {
        self.head
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_none()
    }
}

impl Host<Messages> for ChannelHost {
    async fn project(&self) -> Result<MessagesCall, Error> {
        self.local.project().await
    }

    async fn custom_op(&self, op: Infallible) -> Result<(), Error> {
        match op {}
    }

    async fn open(&self, head: MessagesStreamHead) -> Result<Demand, Error> {
        Ok(match self.take_head().map(|sender| sender.send(head)) {
            Some(Ok(())) => Demand::More,
            Some(Err(_)) | None => Demand::Detached,
        })
    }

    async fn deliver(&self, chunk: Bytes) -> Result<Demand, Error> {
        Ok(match self.chunks.send(chunk).await {
            Ok(()) => Demand::More,
            Err(_) => Demand::Detached,
        })
    }
}
