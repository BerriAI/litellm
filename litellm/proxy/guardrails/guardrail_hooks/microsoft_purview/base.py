import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_ENDPOINT_TEMPLATE = (
    "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
)
GRAPH_SCOPE = "https://graph.microsoft.com/.default"

# Protection scope cache TTL in seconds (1 hour, per Microsoft recommendation).
SCOPE_CACHE_TTL_SECONDS = 3600.0


class PurviewGuardrailBase:
    """
    Base class for Microsoft Purview guardrails.

    Manages OAuth2 client-credentials token acquisition, protection scope
    computation with ETag caching, and authenticated POST calls to the
    Microsoft Graph API.
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        purview_app_name: str = "LiteLLM",
        user_id_field: str = "user_id",
        **kwargs: Any,
    ):
        # Forward remaining kwargs to the next class in the MRO
        # (typically CustomGuardrail).
        super().__init__(**kwargs)

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.purview_app_name = purview_app_name
        self.user_id_field = user_id_field

        # Token cache: (access_token, expires_at_epoch)
        self._token_cache: Optional[Tuple[str, float]] = None

        # Protection scope cache: user_id -> (etag, scope_response, fetched_at)
        # Capped at 1000 entries (LRU eviction) to avoid unbounded growth.
        self._scope_cache: MutableMapping[str, Tuple[str, Dict, float]] = OrderedDict()
        self._scope_cache_maxsize = 1000

    # ------------------------------------------------------------------
    # OAuth2 token management
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """Acquire or return cached OAuth2 token via client_credentials grant."""
        now = time.time()
        if self._token_cache and self._token_cache[1] > now + 60:
            return self._token_cache[0]

        url = TOKEN_ENDPOINT_TEMPLATE.format(tenant_id=self.tenant_id)
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": GRAPH_SCOPE,
        }
        response = await self.async_handler.post(
            url=url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data["access_token"]
        expires_in = int(token_data.get("expires_in", 3599))
        self._token_cache = (access_token, now + expires_in)
        verbose_proxy_logger.debug(
            "Purview: acquired new OAuth2 token (expires_in=%ds)", expires_in
        )
        return access_token

    # ------------------------------------------------------------------
    # Graph API helpers
    # ------------------------------------------------------------------

    async def _graph_post(
        self,
        url: str,
        json_body: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """POST to Graph API with bearer auth.

        Returns:
            Tuple of (response_json, response_headers).
        """
        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        verbose_proxy_logger.debug("Purview Graph POST %s", url)
        response = await self.async_handler.post(
            url=url, headers=headers, json=json_body
        )
        response.raise_for_status()
        response_json: Dict[str, Any] = response.json()
        response_headers = dict(response.headers)
        verbose_proxy_logger.debug("Purview Graph response: %s", response_json)
        return response_json, response_headers

    # ------------------------------------------------------------------
    # Protection scopes
    # ------------------------------------------------------------------

    async def _compute_protection_scopes(
        self, user_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Call protectionScopes/compute and cache with ETag.

        Returns:
            Tuple of (etag, scope_response).
        """
        cached = self._scope_cache.get(user_id)
        now = time.time()

        if cached and (now - cached[2]) < SCOPE_CACHE_TTL_SECONDS:
            return cached[0], cached[1]

        url = (
            f"{GRAPH_API_BASE}/users/{user_id}"
            "/dataSecurityAndGovernance/protectionScopes/compute"
        )
        body: Dict[str, Any] = {
            "activities": "uploadText,downloadText",
            "locations": [
                {
                    "@odata.type": "microsoft.graph.policyLocationApplication",
                    "value": self.client_id,
                }
            ],
        }

        response_json, response_headers = await self._graph_post(url, body)
        etag = response_headers.get("etag", response_headers.get("ETag", ""))

        self._scope_cache[user_id] = (etag, response_json, now)
        # Evict oldest entry when cache exceeds max size.
        while len(self._scope_cache) > self._scope_cache_maxsize:
            self._scope_cache.popitem(last=False)  # type: ignore[attr-defined]
        return etag, response_json

    # ------------------------------------------------------------------
    # Process content
    # ------------------------------------------------------------------

    async def _process_content(
        self,
        user_id: str,
        text: str,
        activity: str,
        etag: str,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call processContent for DLP policy evaluation.

        Args:
            user_id: Entra object ID of the user.
            text: The content to evaluate.
            activity: ``"uploadText"`` for prompts, ``"downloadText"`` for responses.
            etag: Cached ETag from protectionScopes/compute.
            correlation_id: Optional conversation/thread ID.
        """
        url = (
            f"{GRAPH_API_BASE}/users/{user_id}"
            "/dataSecurityAndGovernance/processContent"
        )
        body: Dict[str, Any] = {
            "contentToProcess": {
                "contentEntries": [
                    {
                        "@odata.type": "microsoft.graph.processConversationMetadata",
                        "identifier": str(uuid.uuid4()),
                        "content": {
                            "@odata.type": "microsoft.graph.textContent",
                            "data": text,
                        },
                        "name": f"{self.purview_app_name} message",
                        "correlationId": correlation_id or str(uuid.uuid4()),
                        "sequenceNumber": 0,
                        "isTruncated": False,
                    }
                ],
                "activityMetadata": {"activity": activity},
                "deviceMetadata": {},
                "protectedAppMetadata": {
                    "name": self.purview_app_name,
                    "version": "1.0",
                    "applicationLocation": {
                        "@odata.type": "microsoft.graph.policyLocationApplication",
                        "value": self.client_id,
                    },
                },
                "integratedAppMetadata": {
                    "name": self.purview_app_name,
                    "version": "1.0",
                },
            }
        }

        extra_headers: Dict[str, str] = {}
        if etag:
            extra_headers["If-None-Match"] = etag

        response_json, _ = await self._graph_post(url, body, extra_headers)

        # If policies changed, invalidate scope cache so next call re-fetches.
        if response_json.get("protectionScopeState") == "modified":
            self._scope_cache.pop(user_id, None)

        return response_json

    # ------------------------------------------------------------------
    # User ID resolution
    # ------------------------------------------------------------------

    def _resolve_user_id(
        self, data: Dict[str, Any], user_api_key_dict: Any
    ) -> Optional[str]:
        """Resolve the Entra user object ID from request data or auth context.

        Resolution order:
            1. ``metadata[user_id_field]`` (explicit per-request mapping)
            2. ``user_api_key_dict.user_id``
            3. ``user_api_key_dict.end_user_id``
        """
        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        uid = metadata.get(self.user_id_field)
        if uid:
            return str(uid)
        if hasattr(user_api_key_dict, "user_id") and user_api_key_dict.user_id:
            return str(user_api_key_dict.user_id)
        if hasattr(user_api_key_dict, "end_user_id") and user_api_key_dict.end_user_id:
            return str(user_api_key_dict.end_user_id)
        return None

    # ------------------------------------------------------------------
    # Policy action evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _should_block(response: Dict[str, Any]) -> bool:
        """Return True if any policyAction requires blocking."""
        for action in response.get("policyActions", []):
            odata_type = action.get("@odata.type", "")
            action_field = action.get("action", "")

            if "restrictAccessAction" in odata_type or action_field == "restrictAccess":
                restriction = action.get("restrictionAction", "")
                if restriction == "block":
                    return True
        return False

    # ------------------------------------------------------------------
    # User prompt extraction (same pattern as AzureGuardrailBase)
    # ------------------------------------------------------------------

    def get_user_prompt(self, messages: List["AllMessageValues"]) -> Optional[str]:
        """Get the last consecutive block of user messages as a single string."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_content_list_to_str,
        )

        if not messages:
            return None

        user_messages = []
        for message in reversed(messages):
            if message.get("role") == "user":
                user_messages.append(message)
            else:
                break

        if not user_messages:
            return None

        user_messages.reverse()
        user_prompt = ""
        for message in user_messages:
            text_content = convert_content_list_to_str(message)
            user_prompt += text_content + "\n"

        result = user_prompt.strip()
        return result if result else None
