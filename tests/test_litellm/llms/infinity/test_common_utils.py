"""
#38909: InfinityError's `headers` parameter defaulted to a mutable `{}`,
so every InfinityError raised without an explicit `headers=` argument
(e.g. litellm/llms/infinity/embedding/transformation.py and
litellm/llms/infinity/rerank/transformation.py) shared the exact same
dict object. Mutating one instance's `.headers` would silently leak into
every other default-constructed InfinityError, including ones raised by
unrelated requests.
"""

import httpx

from litellm.llms.infinity.common_utils import InfinityError


class TestInfinityErrorDoesNotShareDefaultHeaders:
    def test_default_headers_is_not_a_mutable_shared_dict(self):
        """The old `headers: dict = {}` default meant every default-constructed
        InfinityError's `.headers` was literally the same dict object, created
        once at function-definition time. The fix defaults to `None` instead,
        so there is nothing shared to mutate."""
        first = InfinityError(status_code=500, message="first failure")

        assert first.headers is None

    def test_two_default_constructed_errors_do_not_share_headers_identity(self):
        """Reproduces the shared-default-object bug directly: with the buggy
        `= {}` default, `first.headers is second.headers` would be True
        because both point at the one dict created when the function was
        defined. Guard against ever reintroducing a mutable default here by
        asserting the default itself is never a dict."""
        first = InfinityError(status_code=500, message="first failure")
        second = InfinityError(status_code=502, message="second failure")

        assert not isinstance(first.headers, dict)
        assert not isinstance(second.headers, dict)

    def test_mutating_a_dict_passed_to_one_instance_does_not_leak_to_another(self):
        """When a caller *does* pass an explicit headers dict, mutating it
        must not affect a different, independently default-constructed
        InfinityError -- this is exactly what happened before the fix,
        since the mutated object was the shared default itself."""
        explicit_headers = {"retry-after": "5"}
        first = InfinityError(status_code=429, message="rate limited", headers=explicit_headers)
        second = InfinityError(status_code=500, message="unrelated failure")

        first.headers["x-request-id"] = "abc-123"

        assert second.headers is None
        assert "x-request-id" not in (second.headers or {})

    def test_explicit_headers_argument_is_still_used_by_identity(self):
        """The fix should not change behavior when a caller *does* pass
        headers explicitly -- the same object should be used as-is."""
        supplied = httpx.Headers({"content-type": "application/json"})
        err = InfinityError(status_code=500, message="failure", headers=supplied)

        assert err.headers is supplied

    def test_call_sites_that_omit_headers_do_not_crash(self):
        """litellm/llms/infinity/embedding/transformation.py and
        litellm/llms/infinity/rerank/transformation.py both raise
        InfinityError without a `headers=` kwarg -- confirm that still
        works and yields None rather than a stale shared dict."""
        err = InfinityError(message="raw response text", status_code=500)

        assert err.status_code == 500
        assert err.message == "raw response text"
        assert err.headers is None
