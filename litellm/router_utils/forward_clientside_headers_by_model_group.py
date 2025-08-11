from typing import Any, Dict, Optional, TypedDict

from litellm.types.utils import CallTypes

from ..integrations.custom_logger import CustomLogger


class PotentialModelGroups(TypedDict):
    deployment_model_name: Optional[str]
    model_group_alias: Optional[str]


class ForwardClientSideHeadersByModelGroup(CustomLogger):
    def get_potential_model_groups_from_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Optional[PotentialModelGroups]:
        """
        Get the model group from the kwargs.

        Returns the potential model groups from the kwargs.
        - deployment_model_name (useful for wildcard model names)
        - model_group_alias (if the model is an alias)
        """
        metadata = kwargs.get("litellm_metadata") or kwargs.get("metadata")
        if metadata is None:
            return None
        deployment_model_name = metadata.get("deployment_model_name", None)
        model_group_alias = metadata.get("model_group_alias", None)
        return {
            "deployment_model_name": deployment_model_name,
            "model_group_alias": model_group_alias,
        }

    def filter_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter the headers to only include the headers that are forwarded to the LLM API.

        E.g. passing 'connection': 'keep-alive' will cause the request to hang, and not be acknowledged on the other side.
        """
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in ["connection", "content-length"]
        }

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

        potential_model_groups = self.get_potential_model_groups_from_kwargs(kwargs)

        if potential_model_groups is None:
            return None

        if (
            "secret_fields" in kwargs
            and kwargs["secret_fields"]["raw_headers"] is not None
            and isinstance(kwargs["secret_fields"]["raw_headers"], dict)
        ):
            for model_group in potential_model_groups.values():
                if model_group is None:
                    continue
                if (
                    litellm.model_group_settings.forward_client_headers_to_llm_api
                    is not None
                    and model_group
                    in litellm.model_group_settings.forward_client_headers_to_llm_api
                ):
                    kwargs.setdefault("headers", {}).update(
                        self.filter_headers(kwargs["secret_fields"]["raw_headers"])
                    )

        return kwargs
