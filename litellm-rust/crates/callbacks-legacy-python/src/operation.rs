use litellm_host::call::Operation;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PassThroughStream {
    pub(crate) url_route: &'static str,
    pub(crate) endpoint_type: &'static str,
}

pub(crate) struct Profile {
    sync_call_type: &'static str,
    async_call_type: &'static str,
    pub(crate) input_description: &'static str,
    pub(crate) stream_billing: Option<PassThroughStream>,
}

impl Profile {
    pub(crate) const fn call_type(&self, asynchronous: bool) -> &'static str {
        if asynchronous {
            self.async_call_type
        } else {
            self.sync_call_type
        }
    }
}

const COMPLETION: Profile = Profile {
    sync_call_type: "completion",
    async_call_type: "acompletion",
    input_description: "Chat completions",
    stream_billing: None,
};

const RESPONSES: Profile = Profile {
    sync_call_type: "responses",
    async_call_type: "aresponses",
    input_description: "Responses",
    stream_billing: None,
};

const MESSAGES: Profile = Profile {
    sync_call_type: "anthropic_messages",
    async_call_type: "anthropic_messages",
    input_description: "Messages",
    stream_billing: Some(PassThroughStream {
        url_route: "/v1/messages",
        endpoint_type: "anthropic",
    }),
};

const OCR: Profile = Profile {
    sync_call_type: "ocr",
    async_call_type: "aocr",
    input_description: "OCR document processing",
    stream_billing: None,
};

pub(crate) const fn profile(operation: Operation) -> &'static Profile {
    match operation {
        Operation::Completion => &COMPLETION,
        Operation::Responses => &RESPONSES,
        Operation::Messages => &MESSAGES,
        Operation::Ocr => &OCR,
    }
}
