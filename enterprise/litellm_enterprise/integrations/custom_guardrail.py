from typing import List, Optional, Union

from litellm.types.guardrails import GuardrailEventHooks, Mode


class EnterpriseCustomGuardrailHelper:
    @staticmethod
    def _should_run_if_mode_by_tag(
        data: dict,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ],
        event_type: Optional[GuardrailEventHooks] = None,
    ) -> Optional[bool]:
        """
        Returns True if the guardrail should be run for this request and event_type.

        Logic:
        - If a request tag matches a Mode tag key, only run if event_type matches
          the tag's value (the mode for that tag).
        - If no request tag matches, fall back to default mode(s).
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

        # Check if any request tag matches a Mode tag key
        matched_mode = None
        if request_tags:
            for tag in request_tags:
                if tag in event_hook.tags:
                    matched_mode = event_hook.tags[tag]
                    break

        if matched_mode is not None:
            # Tag matched: only run if event_type matches the tag's mode value(s)
            if event_type is not None:
                if isinstance(matched_mode, list):
                    return event_type.value in matched_mode
                return event_type.value == matched_mode
            return True

        # No tag matched: fall back to default mode(s)
        if event_hook.default is not None:
            if event_type is not None:
                default_list = (
                    event_hook.default
                    if isinstance(event_hook.default, list)
                    else [event_hook.default]
                )
                return event_type.value in default_list
            return False

        return False
