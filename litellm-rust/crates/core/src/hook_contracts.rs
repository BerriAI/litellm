use serde::{Serialize, de::DeserializeOwned};

pub trait Hook {
    type Input: Serialize + Send + Sync;
    type Output: DeserializeOwned + Send;
}

#[macro_export]
macro_rules! define_hooks {
    (
        $visibility:vis trait $hooks:ident;
        { $($method:ident: $marker:ident($input:ty) -> $output:ty = $mode:ident;)* }
    ) => {
        $(
            $visibility struct $marker;

            impl $crate::hook_contracts::Hook for $marker {
                type Input = $input;
                type Output = $output;
            }
        )*

        $visibility trait $hooks: Send {
            type Error: Send;

            $(
                fn $method<'a>(
                    &'a mut self,
                    input: &'a <$marker as $crate::hook_contracts::Hook>::Input,
                ) -> impl ::std::future::Future<
                    Output = Result<<$marker as $crate::hook_contracts::Hook>::Output, Self::Error>,
                > + Send + 'a;
            )*
        }
    };
}
