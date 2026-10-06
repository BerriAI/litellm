use litellm_cache_response::{CacheKeyContext, CacheKeyField, CacheKeyInput};
use pyo3::{prelude::*, types::PyDict};

fn dictionary<'py>(parent: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    parent
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.cast_into::<PyDict>().map_err(Into::into))
        .unwrap_or_else(|| Ok(PyDict::new(parent.py())))
}

fn text(parent: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    parent
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract())
        .transpose()
}

pub(super) fn project_key(
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<(CacheKeyInput, CacheKeyContext)> {
    let parameters = dictionary(arguments, "litellm_params")?;
    if let Some(preset) = text(&parameters, "preset_cache_key")? {
        return Ok((
            CacheKeyInput {
                preset: Some(preset),
                ..Default::default()
            },
            CacheKeyContext::default(),
        ));
    }
    let metadata = dictionary(arguments, "metadata")?;
    let sources = [
        metadata.clone(),
        dictionary(arguments, "litellm_metadata")?,
        dictionary(&parameters, "metadata")?,
        dictionary(&parameters, "litellm_metadata")?,
    ];
    let model_group = sources
        .iter()
        .map(|source| text(source, "model_group"))
        .find(|result| match result {
            Err(_) => true,
            Ok(value) => value.as_ref().is_some_and(|value| !value.is_empty()),
        })
        .transpose()?
        .flatten();
    let groups = sources
        .iter()
        .map(|source| {
            source
                .get_item("caching_groups")?
                .filter(|value| !value.is_none())
                .map(|value| {
                    value
                        .try_iter()?
                        .map(|group| {
                            let group = group?;
                            Ok((group.extract::<Vec<String>>()?, group.str()?.to_string()))
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
                .transpose()
        })
        .collect::<PyResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let file_checksum = text(&metadata, "file_checksum")?.filter(|value| !value.is_empty());
    let file_object_name = if file_checksum.is_some() {
        None
    } else {
        arguments
            .get_item("file")?
            .filter(|file| !file.is_none())
            .map(|file| {
                file.getattr_opt("name")?
                    .map(|name| name.extract())
                    .transpose()
            })
            .transpose()?
            .flatten()
    };
    let context = CacheKeyContext {
        model_group,
        caching_groups: groups,
        file_checksum,
        file_object_name,
        metadata_file_name: text(&metadata, "file_name")?,
        parameters_file_name: text(&parameters, "file_name")?,
    };
    let api_parameters = py
        .import("litellm.litellm_core_utils.model_param_helper")?
        .getattr("ModelParamHelper")?
        .call_method0("_get_all_llm_api_params")?;
    let owned = py
        .import("litellm.types.utils")?
        .getattr("is_litellm_owned_kwarg")?;
    let include_provider_parameters: bool = py
        .import("litellm")?
        .getattr("enable_caching_on_provider_specific_optional_params")?
        .extract()?;
    let fields = arguments
        .iter()
        .map(|(name, value)| {
            let name: String = name.extract()?;
            let api_parameter = api_parameters.contains(&name)?;
            let internal_parameter = owned.call1((&name,))?.extract()?;
            let encoded = if name == "file"
                || value.is_none()
                || (!api_parameter && (!include_provider_parameters || internal_parameter))
            {
                None
            } else {
                Some(value.str()?.to_string())
            };
            Ok(CacheKeyField {
                name,
                value: encoded,
                api_parameter,
                internal_parameter,
            })
        })
        .collect::<PyResult<_>>()?;
    let controls = dictionary(arguments, "cache")?;
    let namespace = text(&controls, "namespace")?
        .filter(|value| !value.is_empty())
        .or(text(&metadata, "redis_namespace")?.filter(|value| !value.is_empty()));
    Ok((
        CacheKeyInput {
            fields,
            preset: text(&parameters, "preset_cache_key")?,
            namespace,
            include_provider_parameters,
            ..Default::default()
        },
        context,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_cache_response::cache_key;
    use rstest::{fixture, rstest};
    use std::ffi::CString;

    #[fixture]
    fn interpreter() {
        Python::initialize();
    }

    #[rstest]
    #[case::metadata("metadata={'model_group':'logical'}")]
    #[case::alternate_metadata("litellm_metadata={'model_group':'logical'}")]
    #[case::nested_metadata("litellm_params={'metadata':{'model_group':'logical'}}")]
    #[case::nested_alternate("litellm_params={'litellm_metadata':{'model_group':'logical'}}")]
    #[case::precedence(
        "metadata={'model_group':'first'}, litellm_metadata={'model_group':'second'}"
    )]
    #[case::empty_group("metadata={'model_group':''}, litellm_metadata={'model_group':'second'}")]
    #[case::cross_source_group(
        "metadata={'model_group':'logical'}, litellm_metadata={'caching_groups':[['logical','other']]} "
    )]
    #[case::namespace("cache={'namespace':'override'}, metadata={'redis_namespace':'fallback'}")]
    #[case::preset("litellm_params={'preset_cache_key':'verbatim'}, cache={'namespace':'ignored'}")]
    #[case::file_checksum(
        "file=object(), metadata={'file_checksum':'checksum','file_name':'ignored'}"
    )]
    #[case::checksum_precedes_file_name(
        "file=type('File', (), {'name':property(lambda self: 1/0)})(), metadata={'file_checksum':'stable'}"
    )]
    #[case::file_object_name(
        "file=type('File', (), {'name':'stable', '__str__':lambda self: 1/0})()"
    )]
    #[case::file_name("file=object(), litellm_params={'file_name':'fallback'}")]
    #[case::policy("cache={'ttl':7,'no-cache':True,'no-store':True}")]
    #[case::provider_parameter("custom_parameter={'enabled':True}")]
    fn projected_key_matches_python(
        _interpreter: (),
        #[case] extra: &str,
        #[values(false, true)] include_provider: bool,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let source = CString::new(format!(
                "from litellm.caching.caching import Cache\n\
                 arguments = dict(model='deployment', messages=[{{'role':'user','content':'hello \\u263a'}}], temperature=0.5, stream=False, {extra})\n\
                 cache = Cache.__new__(Cache)\n\
                 cache.namespace = 'default'\n\
                 cache.type = 'local'\n"
            )).unwrap();
            py.run(&source, None, Some(&locals)).unwrap();
            let arguments = locals
                .get_item("arguments")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let litellm = py.import("litellm").unwrap();
            let previous = litellm
                .getattr("enable_caching_on_provider_specific_optional_params")
                .unwrap();
            litellm
                .setattr(
                    "enable_caching_on_provider_specific_optional_params",
                    include_provider,
                )
                .unwrap();
            let (mut input, context) = project_key(py, &arguments).unwrap();
            context.apply(&mut input);
            input.namespace = input.namespace.or(Some("default".into()));
            let expected: String = locals
                .get_item("cache")
                .unwrap()
                .unwrap()
                .call_method("get_cache_key", (), Some(&arguments))
                .unwrap()
                .extract()
                .unwrap();
            litellm
                .setattr(
                    "enable_caching_on_provider_specific_optional_params",
                    previous,
                )
                .unwrap();
            assert_eq!(cache_key(&input), expected);
        });
    }
}
