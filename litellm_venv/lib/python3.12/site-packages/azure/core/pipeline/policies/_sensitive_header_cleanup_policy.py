# --------------------------------------------------------------------------
#
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the ""Software""), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# --------------------------------------------------------------------------
from typing import List, Optional, Any, TypeVar
from azure.core.pipeline import PipelineRequest
from azure.core.pipeline.transport import (
    HttpRequest as LegacyHttpRequest,
    HttpResponse as LegacyHttpResponse,
)
from azure.core.rest import HttpRequest, HttpResponse
from ._base import SansIOHTTPPolicy

HTTPResponseType = TypeVar("HTTPResponseType", HttpResponse, LegacyHttpResponse)
HTTPRequestType = TypeVar("HTTPRequestType", HttpRequest, LegacyHttpRequest)


class SensitiveHeaderCleanupPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A simple policy that cleans up sensitive headers

    :keyword list[str] blocked_redirect_headers: The headers to clean up when redirecting to another domain.
    :keyword bool disable_redirect_cleanup: Opt out cleaning up sensitive headers when redirecting to another domain.
    """

    DEFAULT_SENSITIVE_HEADERS = set(
        [
            "Authorization",
            "x-ms-authorization-auxiliary",
        ]
    )

    def __init__(
        self,  # pylint: disable=unused-argument
        *,
        blocked_redirect_headers: Optional[List[str]] = None,
        disable_redirect_cleanup: bool = False,
        **kwargs: Any
    ) -> None:
        self._disable_redirect_cleanup = disable_redirect_cleanup
        self._blocked_redirect_headers = (
            SensitiveHeaderCleanupPolicy.DEFAULT_SENSITIVE_HEADERS
            if blocked_redirect_headers is None
            else blocked_redirect_headers
        )

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        """This is executed before sending the request to the next policy.

        :param request: The PipelineRequest object.
        :type request: ~azure.core.pipeline.PipelineRequest
        """
        # "insecure_domain_change" is used to indicate that a redirect
        # has occurred to a different domain. This tells the SensitiveHeaderCleanupPolicy
        # to clean up sensitive headers. We need to remove it before sending the request
        # to the transport layer.
        insecure_domain_change = request.context.options.pop(
            "insecure_domain_change", False
        )
        if not self._disable_redirect_cleanup and insecure_domain_change:
            for header in self._blocked_redirect_headers:
                request.http_request.headers.pop(header, None)
