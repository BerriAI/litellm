import json
import os
import traceback
import types

import requests


class LiteDebugger:
    user_email = None
    dashboard_url = None

    def __init__(self, email=None):
        self.api_url = "https://api.litellm.ai/debugger"
        self.validate_environment(email)
        pass

    def validate_environment(self, email):
        try:
            self.user_email = (
                email or os.getenv("LITELLM_TOKEN") or os.getenv("LITELLM_EMAIL")
            )
            if (
                self.user_email is None
            ):  # if users are trying to use_client=True but token not set
                raise ValueError(
                    "litellm.use_client = True but no token or email passed. Please set it in litellm.token"
                )
            self.dashboard_url = "https://admin.litellm.ai/" + self.user_email
            if self.user_email is None:
                raise ValueError(
                    "[Non-Blocking Error] LiteLLMDebugger: Missing LITELLM_TOKEN. Set it in your environment. Eg.: os.environ['LITELLM_TOKEN']= <your_email>"
                )
        except Exception:
            raise ValueError(
                "[Non-Blocking Error] LiteLLMDebugger: Missing LITELLM_TOKEN. Set it in your environment. Eg.: os.environ['LITELLM_TOKEN']= <your_email>"
            )

    def input_log_event(
        self,
        model,
        messages,
        end_user,
        litellm_call_id,
        call_type,
        print_verbose,
        litellm_params,
        optional_params,
    ):
        """
        This integration is not implemented yet.
        """
        return

    def post_call_log_event(
        self, original_response, litellm_call_id, print_verbose, call_type, stream
    ):
        """
        This integration is not implemented yet.
        """
        return

    def log_event(
        self,
        end_user,
        response_obj,
        start_time,
        end_time,
        litellm_call_id,
        print_verbose,
        call_type,
        stream=False,
    ):
        """
        This integration is not implemented yet.
        """
        return
