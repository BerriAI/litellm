"""
GDC Gemini chat completion transformation
"""

import litellm
from typing import Any, List, Optional
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class GDCGeminiConfig(OpenAILikeChatConfig):

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = (
            api_base or litellm.api_base or getattr(litellm, "gdc_api_base", None)
        )
        if not api_base:
            raise litellm.utils.AuthenticationError(
                message="api_base/host is required for GDC Gemini. Please set it or pass it.",
                llm_provider="gdc",
                model=model,
            )

        if not api_base.startswith("http"):
            api_base = f"https://{api_base}"

        project = (
            optional_params.get("vertex_project")
            or litellm_params.get("vertex_project")
            or getattr(litellm, "vertex_project", None)
        )
        if not project:
            raise litellm.utils.AuthenticationError(
                message="project is required for GDC Gemini. Please pass vertex_project.",
                llm_provider="gdc",
                model=model,
            )

        api_base = api_base.rstrip("/")

        # If the endpoint structure is already in the api_base, don't append it again
        if "/v1/projects/" in api_base:
            base_url = api_base
        else:
            base_url = f"{api_base}/v1/projects/{project}/locations/{project}"

        endpoint = "chat/completions"
        if "chat/completions" in base_url:
            return base_url

        return f"{base_url}/{endpoint}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        import json
        import google.auth
        import requests
        from google.auth.transport import requests as auth_requests

        # Extract GDC API base
        api_base = (
            api_base or litellm.api_base or getattr(litellm, "gdc_api_base", None)
        )

        if not api_base:
            raise litellm.utils.AuthenticationError(
                message="api_base/host is required for GDC Gemini. Please set it or pass it.",
                llm_provider="gdc",
                model=model,
            )

        if not api_key:
            raise litellm.utils.AuthenticationError(
                message="api_key is required for GDC Gemini. Please pass your service account string or token as the api_key.",
                llm_provider="gdc",
                model=model,
            )

        project = (
            optional_params.get("vertex_project")
            or litellm_params.get("vertex_project")
            or getattr(litellm, "vertex_project", None)
        )
        if not project:
            raise litellm.utils.AuthenticationError(
                message="project is required for GDC Gemini. Please pass vertex_project.",
                llm_provider="gdc",
                model=model,
            )

        # Ensure we have the audience for token fetch
        audience = api_base
        if not audience.startswith("http"):
            audience = f"https://{audience}"

        # Generate GDC token
        is_service_account = False
        try:
            creds = None
            if api_key:
                import os

                try:
                    # Check if api_key is a file path
                    # Limit length to avoid OSError for 'File name too long'
                    if len(api_key) < 2000 and os.path.exists(api_key):
                        with open(api_key, "r") as f:
                            json_obj = json.load(f)

                        is_service_account = True
                    else:
                        json_obj = json.loads(api_key)
                        is_service_account = True

                    if is_service_account:
                        creds, _ = google.auth.load_credentials_from_dict(json_obj)
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    # It's not a valid JSON string or file path, treat as raw token
                    is_service_account = False
                except Exception as e:
                    raise litellm.utils.AuthenticationError(
                        message=f"Failed to load service account credentials from api_key: {str(e)}",
                        llm_provider="gdc",
                        model=model,
                    )

            if creds is not None:
                # if hasattr(creds, "with_gdch_audience"):
                creds = creds.with_gdch_audience(audience)
                auth_session = requests.Session()

                # Retrieve ssl_verify configuration
                ssl_verify = litellm_params.get("ssl_verify")
                if ssl_verify is None:
                    import os

                    ssl_verify = os.getenv("SSL_VERIFY", True)
                    if isinstance(ssl_verify, str):
                        if ssl_verify.lower() == "false":
                            ssl_verify = False
                        elif ssl_verify.lower() == "true":
                            ssl_verify = True

                auth_session.verify = ssl_verify
                auth_request = auth_requests.Request(session=auth_session)

                creds.refresh(auth_request)
                token = creds.token
                headers["Authorization"] = f"Bearer {token}"
        except Exception as e:
            raise e

        if "Authorization" not in headers and api_key and not is_service_account:
            headers["Authorization"] = f"Bearer {api_key}"

        # Ensure Content-Type is set to application/json
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        # Ensure Vertex Project is included in headers
        if (
            "x-goog-user-project" not in headers
            and "X-Goog-User_project" not in headers
        ):
            headers["x-goog-user-project"] = f"projects/{project}"

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transforms the request to the GDC provider
        """
        # Strip provider prefix for GDC
        if model.startswith("gdc/"):
            model = model.split("/", 1)[1]

        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Remove extra params used for routing/auth
        data.pop("vertex_project", None)
        data.pop("ssl_verify", None)

        return data
