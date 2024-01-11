import pytest
from litellm import acompletion


def test_acompletion_params():
    import inspect
    from litellm.types.completion import CompletionRequest

    acompletion_params_odict = inspect.signature(acompletion).parameters
    acompletion_params = {name: param.annotation for name, param in acompletion_params_odict.items()}
    completion_params = {field_name: field_type for field_name, field_type in CompletionRequest.__annotations__.items()}

    # remove kwargs
    acompletion_params.pop("kwargs", None)

    keys_acompletion = set(acompletion_params.keys())
    keys_completion = set(completion_params.keys())

    # Assert that the parameters are the same
    if keys_acompletion != keys_completion:
        pytest.fail("The parameters of the acompletion function and the CompletionRequest class are not the same.")

# test_acompletion_params()
