#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackBodyView {
    Structured,
    Serialized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyReadPoint {
    BuildRequest,
    Send,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestBodyBehavior {
    pub callback_view: CallbackBodyView,
    pub transport_read: BodyReadPoint,
}

impl RequestBodyBehavior {
    pub const STRUCTURED_AT_SEND: Self = Self {
        callback_view: CallbackBodyView::Structured,
        transport_read: BodyReadPoint::Send,
    };

    pub const STRUCTURED_AT_BUILD: Self = Self {
        callback_view: CallbackBodyView::Structured,
        transport_read: BodyReadPoint::BuildRequest,
    };

    pub const SERIALIZED_AT_BUILD: Self = Self {
        callback_view: CallbackBodyView::Serialized,
        transport_read: BodyReadPoint::BuildRequest,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_view_and_transport_read_are_independent() {
        assert_eq!(
            RequestBodyBehavior::STRUCTURED_AT_BUILD,
            RequestBodyBehavior {
                callback_view: CallbackBodyView::Structured,
                transport_read: BodyReadPoint::BuildRequest,
            }
        );
        assert_ne!(
            RequestBodyBehavior::STRUCTURED_AT_BUILD,
            RequestBodyBehavior::SERIALIZED_AT_BUILD
        );
    }
}
