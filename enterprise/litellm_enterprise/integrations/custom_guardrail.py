from typing import List, Optional, Union

from litellm.types.guardrails import GuardrailEventHooks, Mode


class EnterpriseCustomGuardrailHelper:
    @staticmethod
    def _should_run_if_mode_by_tag(
        data: dict,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ],
    ) -> Optional[bool]:
        """
        Assumes check for event match is done in `should_run_guardrail`
        Returns True if the guardrail should be run by tag
        """
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )
        from litellm.proxy._types import CommonProxyErrors
        from litellm.proxy.proxy_server import premium_user

        if not premium_user:
            raise Exception(
                f"Setting tag based guardrail modes is only available in litellm-enterprise. {CommonProxyErrors.not_premium_user.value}."
            )

        if event_hook is None or not isinstance(event_hook, Mode):
            return None

        proxy_server_request = data.get("proxy_server_request", {})

        request_tags = StandardLoggingPayloadSetup._get_request_tags(
            litellm_params=data,
            proxy_server_request=proxy_server_request,
        )

        if request_tags and any(tag in event_hook.tags for tag in request_tags):
            return True
        elif event_hook.default and any(
            tag in event_hook.default for tag in request_tags
        ):
            return True

        return False
