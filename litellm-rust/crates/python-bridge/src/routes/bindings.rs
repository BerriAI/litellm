use litellm_python_interop::InvocationMode;

pub(crate) struct MethodBinding {
    pub(crate) name: &'static str,
    pub(crate) mode: InvocationMode,
}

pub(crate) enum BoundaryMethod {
    Prepare,
    Encode,
    Finish,
}

impl BoundaryMethod {
    pub(crate) fn resolve(self, asynchronous: bool) -> MethodBinding {
        match (self, asynchronous) {
            (Self::Prepare, true) => MethodBinding {
                name: "aprepare",
                mode: InvocationMode::Await,
            },
            (Self::Prepare, false) => MethodBinding {
                name: "prepare",
                mode: InvocationMode::Direct,
            },
            (Self::Encode, _) => MethodBinding {
                name: "encode",
                mode: InvocationMode::Direct,
            },
            (Self::Finish, true) => MethodBinding {
                name: "afinish",
                mode: InvocationMode::Await,
            },
            (Self::Finish, false) => MethodBinding {
                name: "finish",
                mode: InvocationMode::Direct,
            },
        }
    }
}
