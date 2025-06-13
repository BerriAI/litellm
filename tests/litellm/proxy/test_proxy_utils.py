from unittest.mock import MagicMock, patch

import pytest

import litellm


@pytest.mark.parametrize("api_base, v1_models_url", [("http://foo.bar/baz", "http://foo.bar/baz/v1/models"),
                                                     ("http://foo.bar/v1", "http://foo.bar/v1/models")])
@pytest.mark.parametrize("return_wildcard_routes", [True, False
                                                    ])
@pytest.mark.parametrize("deployment", ["openai",
                                        # "litellm_proxy" is out of scope since it prepends models with own name litellm.llms.litellm_proxy.chat.transformation.LiteLLMProxyChatConfig.get_models
                                        ])
def test_get_complete_model_list_wildcard_models(api_base: str, v1_models_url: str,
                                                 return_wildcard_routes: bool,
                                                 deployment):
    from litellm.proxy.auth.model_checks import (get_complete_model_list)
    # Create a mock response object
    model_name = "llama645"
    provider_prefix = "foo"
    top_secret = "top_secret"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
        ],
    }
    from litellm.utils import AvailableModelsCache
    with (patch('litellm.check_provider_endpoint', new=True),
          patch('litellm.utils._model_cache', new=AvailableModelsCache()),
          patch.object(
              litellm.module_level_client, "get", return_value=mock_response
          ) as mock_get):
        from litellm import Router
        model_list = [
            {
                "model_name": provider_prefix + "/*",  # model alias
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": deployment + "/*",
                    "api_key": top_secret,
                    "api_base": api_base,
                }
            }
        ]

        wildcard_models = get_complete_model_list(key_models=[provider_prefix + "/*"],
                                                  team_models=[], proxy_model_list=[], user_model=None,
                                                  infer_model_from_keys=False,
                                                  return_wildcard_routes=return_wildcard_routes,
                                                  llm_router=Router(model_list=model_list))
        assert len(wildcard_models) == len(set(wildcard_models))
        assert set(wildcard_models) == set(
            [provider_prefix + "/" + model_name] + ([provider_prefix + "/*"] if return_wildcard_routes else []))
        mock_get.assert_called_once_with(url=v1_models_url,
                                         headers={"Authorization": f"Bearer {top_secret}"})
