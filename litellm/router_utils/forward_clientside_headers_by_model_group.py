from typing import Any, Dict, Optional

from litellm.types.utils import CallTypes

from ..integrations.custom_logger import CustomLogger


class ForwardClientSideHeadersByModelGroup(CustomLogger):
    def get_model_group_from_kwargs(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Get the model group from the kwargs.
        """
        metadata = kwargs.get("litellm_metadata") or kwargs.get("metadata")
        if metadata is None:
            return None
        return metadata.get("model_group", None)

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        if kwargs["proxy_server_request"]["headers"] is not None:
        and kwargs["forward_client_headers_to_llm_api"] is not None:

        add the headers to the request
        kwargs["headers"].update(kwargs["proxy_server_request"]["headers"])
        """
        import litellm

        if litellm.model_group_settings is None:
            return None

        model_group = self.get_model_group_from_kwargs(kwargs)

        if model_group is None:
            return None

        if (
            "secret_fields" in kwargs
            and kwargs["secret_fields"]["raw_headers"] is not None
            and isinstance(kwargs["secret_fields"]["raw_headers"], dict)
        ):
            if (
                litellm.model_group_settings.forward_client_headers_to_llm_api
                is not None
                and model_group
                in litellm.model_group_settings.forward_client_headers_to_llm_api
            ):
                kwargs.setdefault("headers", {}).update(
                    kwargs["secret_fields"]["raw_headers"]
                )

        return kwargs
