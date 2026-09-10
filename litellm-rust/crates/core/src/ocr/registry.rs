use super::adapters::{AdapterConfig, InputParams, OcrAdapter};
use crate::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};
use serde_json::{Map, Value};

#[derive(Clone, Debug)]
pub(crate) struct OcrAdapterInput<A: OcrAdapter> {
    pub params: InputParams<A>,
    pub config: AdapterConfig<A>,
}

macro_rules! define_adapter_types {
    ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
        #[derive(Clone, Debug)]
        pub(crate) enum OcrAdapterRequest {
            $( $variant(OcrAdapterInput<$adapter>), )+
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum OcrAdapterKind {
            $( $variant, )+
        }

        impl OcrAdapterKind {
            pub(crate) const fn provider(self) -> OcrProvider {
                match self {
                    $( Self::$variant => OcrProvider::$provider, )+
                }
            }
        }

        impl OcrAdapterRequest {
            pub(crate) fn kind(&self) -> OcrAdapterKind {
                match self {
                    $( Self::$variant(_) => OcrAdapterKind::$variant, )+
                }
            }
        }
    };
}

super::adapters::for_each_ocr_adapter!(define_adapter_types);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrProvider {
    Mistral,
}

impl OcrProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
        }
    }
}

pub(crate) fn resolve_wire_adapter(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(String, OcrAdapterKind), Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.as_str(),
        });
    let typed_provider = match provider.custom_llm_provider {
        "mistral" => OcrProvider::Mistral,
        value => return Err(Error::InvalidProvider(value.to_string())),
    };
    match typed_provider {
        OcrProvider::Mistral => Ok((provider.model.to_string(), OcrAdapterKind::Mistral)),
    }
}

pub(crate) fn decode_adapter_request(
    kind: OcrAdapterKind,
    params: Map<String, Value>,
) -> Result<OcrAdapterRequest, Error> {
    macro_rules! decode_selected_adapter {
        ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
            match kind {
                $(
                    OcrAdapterKind::$variant => OcrAdapterRequest::$variant(
                        decode_input::<$adapter>($instance, params)?,
                    ),
                )+
            }
        };
    }

    Ok(super::adapters::for_each_ocr_adapter!(
        decode_selected_adapter
    ))
}

fn decode_input<A>(adapter: A, params: Map<String, Value>) -> Result<OcrAdapterInput<A>, Error>
where
    A: OcrAdapter,
{
    let config = A::decode_config(&params)?;
    let params = adapter.decode_input_params(params, "optional_params")?;
    Ok(OcrAdapterInput { params, config })
}
