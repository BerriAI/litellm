from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.watsonx.common_utils import IBMWatsonXMixin
from litellm.types.passthrough_endpoints.watsonx import WatsonxPassThroughCredentials

if TYPE_CHECKING:
    from httpx import URL

    from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict


class WatsonxPassthroughConfig(IBMWatsonXMixin, BasePassthroughConfig):
    """
    Watsonx-specific passthrough configuration.
    """

    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """Check if request should be streamed"""
        return request_data.get("stream", False)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict | None,
        litellm_params: dict,
    ) -> tuple["URL", str]:
        """
        Construct complete Watsonx URL with version parameter.

        This ensures the version parameter is ALWAYS included in the URL,
        solving the query parameter issue.
        """
        base_target_url: Final = str(self.get_api_base(api_base))

        # Use the format_url helper to construct URL with query params
        complete_url: Final = self.format_url(
            endpoint=endpoint,
            base_target_url=base_target_url,
            request_query_params=request_query_params,
        )

        return (complete_url, base_target_url)

    @staticmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str | None:
        return api_base or IBMWatsonXMixin()._get_base_url(api_base=api_base)

    @staticmethod
    def get_api_key(
        api_key: str | None = None,
    ) -> str | None:
        return (
            api_key
            or IBMWatsonXMixin.get_watsonx_credentials(optional_params=dict(), api_base=None, api_key=api_key)[
                "api_key"
            ]
        )

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return super().get_models(api_key, api_base)

    # ------------------------------------------------------------------
    # Router deployment credential resolution
    # ------------------------------------------------------------------

    @classmethod
    def resolve_credentials_from_router_deployments(
        cls,
        model_id: str | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        deployments: "Sequence[DeploymentTypedDict] | None" = None,
    ) -> "WatsonxPassThroughCredentials | None":
        """
        Resolve watsonx pass-through credentials from router deployments flagged
        with ``use_in_pass_through: true``.

        Disambiguation rules (applied in order):
        1. Exact match: model_id + project_id (or space_id) — both must match.
        2. Partial match: model_id only — first deployment whose model matches.
        3. Fallback: first eligible deployment (warns if multiple exist).

        Returns ``None`` when no eligible deployment exists, allowing the caller
        to continue with env-var resolution.
        """
        candidates: Final = tuple(
            creds
            for deployment in (deployments or ())
            if (creds := cls.resolve_credentials_from_deployment(deployment["litellm_params"])) is not None
        )

        if not candidates:
            return None
        first_candidate: Final = candidates[0]

        def model_ok(creds: WatsonxPassThroughCredentials) -> bool:
            return bool(model_id and cls.deployment_model_matches(creds, model_id))

        def project_ok(creds: WatsonxPassThroughCredentials) -> bool:
            return bool(project_id) and creds.get("project_id") == project_id

        def space_ok(creds: WatsonxPassThroughCredentials) -> bool:
            return bool(space_id) and creds.get("space_id") == space_id

        selection_rules: Final[tuple[Callable[[WatsonxPassThroughCredentials], bool], ...]] = (
            lambda creds: model_ok(creds) and project_ok(creds),
            lambda creds: model_ok(creds) and space_ok(creds),
            model_ok,
            project_ok,
            space_ok,
        )
        matched: Final = next(
            filter(None, (next(filter(rule, candidates), None) for rule in selection_rules)),
            None,
        )
        if matched is not None:
            return matched

        # Fallback: single deployment — unambiguous; multiple — return first with a warning
        if len(candidates) > 1:
            verbose_router_logger.warning(
                "Multiple watsonx use_in_pass_through deployments found but no "
                "model_id/project_id/space_id supplied to disambiguate — using the first. "
                "Set project_id or space_id on each deployment's litellm_params and pass "
                "project_id or space_id in the request body to route to the correct credentials."
            )
        return first_candidate

    @staticmethod
    def deployment_model_matches(creds: "WatsonxPassThroughCredentials", model_id: str) -> bool:
        """Check if the credential's stored model ends with the request's model_id."""
        stored: Final = creds.get("model")
        if stored is None:
            return False
        return stored == model_id or str(stored).endswith(f"/{model_id}")

    @classmethod
    def resolve_credentials_from_deployment(
        cls, litellm_params: "LiteLLMParamsTypedDict"
    ) -> "WatsonxPassThroughCredentials | None":
        """
        Extract WatsonxPassThroughCredentials from a single deployment's litellm_params
        if it is flagged ``use_in_pass_through: true`` and is a watsonx provider.
        Returns ``None`` for non-watsonx or unflagged deployments.
        """
        if litellm_params.get("use_in_pass_through") is not True:
            return None
        model: Final = litellm_params.get("model")
        if model is None:
            return None
        try:
            _, provider, _, _ = get_llm_provider(
                model=model,
                custom_llm_provider=litellm_params.get("custom_llm_provider"),
            )
        except BadRequestError:
            return None
        if provider != "watsonx":
            return None
        credential_name: Final = litellm_params.get("litellm_credential_name")
        credential_values: Final[Mapping[str, object]] = (
            CredentialAccessor.get_credential_values(credential_name) if credential_name is not None else {}
        )
        credential_api_key: Final = credential_values.get("api_key")
        credential_api_base: Final = credential_values.get("api_base")
        api_key: Final = (credential_api_key if isinstance(credential_api_key, str) else None) or litellm_params.get(
            "api_key"
        )
        api_base: Final = (credential_api_base if isinstance(credential_api_base, str) else None) or litellm_params.get(
            "api_base"
        )
        # api_key and api_base are both optional: either can be None and fall back to
        # WATSONX_API_KEY / WATSONX_API_BASE env vars inside validate_environment / get_complete_url.
        return WatsonxPassThroughCredentials(
            api_key=api_key,
            api_base=api_base,
            project_id=litellm_params.get("project_id"),
            space_id=litellm_params.get("space_id"),
            model=model,
        )
