#[macro_export]
macro_rules! callback_return_mode {
    (direct) => {
        ::litellm_python_interop::callback_runtime::Direct
    };
    (awaitable) => {
        ::litellm_python_interop::callback_runtime::Awaitable
    };
}

#[macro_export]
macro_rules! bind_python_hooks {
    (
        $visibility:vis struct $session:ident;
        trait $hooks:path;
        { $($method:ident: $marker:ident($input:ty) -> $output:ty = $mode:ident;)* }
    ) => {
        $visibility struct $session<C> {
            context: C,
            $(
                $method: ::litellm_python_interop::callback_runtime::Callback<
                    $input, $output, $crate::callback_return_mode!($mode),
                >,
            )*
        }

        impl<C> $session<C>
        where
            $(C: ::litellm_python_interop::callback_runtime::CallbackContext<
                $crate::callback_return_mode!($mode),
            >,)*
        {
            $visibility fn new(
                adapter: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                context: C,
            ) -> ::pyo3::PyResult<Self> {
                use ::pyo3::types::PyAnyMethods as _;
                Ok(Self {
                    context,
                    $(
                        $method: ::litellm_python_interop::callback_runtime::Callback::new(
                            adapter.getattr(stringify!($method))?,
                        )?,
                    )*
                })
            }
        }

        impl<C> $hooks for $session<C>
        where
            $(C: ::litellm_python_interop::callback_runtime::CallbackContext<
                $crate::callback_return_mode!($mode),
            >,)*
        {
            type Error = ::pyo3::PyErr;

            $(
                async fn $method(&mut self, input: &$input) -> ::pyo3::PyResult<$output> {
                    self.$method.call(&mut self.context, input).await
                }
            )*
        }
    };
}
