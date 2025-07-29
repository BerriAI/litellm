from typing import Dict, List, Optional, Union
from urllib.parse import parse_qs

import httpx


class BasePassthroughUtils:
    @staticmethod
    def get_merged_query_parameters(
        existing_url: httpx.URL, request_query_params: Dict[str, Union[str, list]]
    ) -> Dict[str, Union[str, List[str]]]:
        # Get the existing query params from the target URL
        existing_query_string = existing_url.query.decode("utf-8")
        existing_query_params = parse_qs(existing_query_string)

        # parse_qs returns a dict where each value is a list, so let's flatten it
        updated_existing_query_params = {
            k: v[0] if len(v) == 1 else v for k, v in existing_query_params.items()
        }
        # Merge the query params, giving priority to the existing ones
        return {**request_query_params, **updated_existing_query_params}

    @staticmethod
    def forward_headers_from_request(
        request_headers: dict,
        headers: dict,
        forward_headers: Optional[bool] = False,
    ):
        """
        Helper to forward headers from original request
        """
        if forward_headers is True:
            # Header We Should NOT forward
            request_headers.pop("content-length", None)
            request_headers.pop("host", None)

            # Combine request headers with custom headers
            headers = {**request_headers, **headers}
        return headers
