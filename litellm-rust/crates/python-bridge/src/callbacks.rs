use pyo3::prelude::*;

#[derive(IntoPyObject)]
pub(crate) struct AdditionalArgs<Body, Base, Headers> {
    complete_input_dict: Body,
    api_base: Base,
    headers: Headers,
}

#[derive(IntoPyObject)]
pub(crate) struct PreCallArgs<Input, Key, Additional> {
    input: Input,
    api_key: Key,
    additional_args: Additional,
}

pub(crate) fn pre_call_args<Input, Key, Body, Base, Headers>(
    input: Input,
    api_key: Key,
    body: Body,
    api_base: Base,
    headers: Headers,
) -> PreCallArgs<Input, Key, AdditionalArgs<Body, Base, Headers>> {
    PreCallArgs {
        input,
        api_key,
        additional_args: AdditionalArgs {
            complete_input_dict: body,
            api_base,
            headers,
        },
    }
}
