//! The `litellm_proxy.config_change` channel the proxy's writers publish on after each write to a
//! config table or settings row (`config_sync_pubsub.py`), heard the way `ConfigSyncSubscriber`
//! hears it. The gateway only listens, so nothing here publishes.

mod keys;
mod types;

use futures_util::{StreamExt, stream::BoxStream};
use litellm_redis_client::{RedisClient, Subscription};

pub use self::keys::channel;
pub use self::types::ConfigChange;

/// What a subscriber to the channel hears. `Lost` and `Subscribed` come in that order around every
/// stretch without a subscription, during which published changes were missed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Heard {
    Subscribed,
    Lost,
    Change(ConfigChange),
}

/// Every change published on the channel `namespace` prefixes, on its own connection to `client`'s
/// Redis, which resubscribes after a lost connection. A message that does not decode is skipped.
pub fn subscribe(client: &RedisClient, namespace: Option<&str>) -> BoxStream<'static, Heard> {
    client
        .subscription(&channel(namespace))
        .filter_map(|event| {
            std::future::ready(match event {
                Subscription::Subscribed => Some(Heard::Subscribed),
                Subscription::Lost => Some(Heard::Lost),
                Subscription::Message(payload) => {
                    let change = ConfigChange::decode(&payload).map(Heard::Change);
                    if change.is_none() {
                        tracing::debug!(
                            payload = %String::from_utf8_lossy(&payload),
                            "ignoring a config change message that does not decode"
                        );
                    }
                    change
                }
            })
        })
        .boxed()
}
