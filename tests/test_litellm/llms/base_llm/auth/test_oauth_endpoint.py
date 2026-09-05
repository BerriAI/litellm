import base64
import json
from typing import Final
from urllib.parse import quote, urlencode

import pytest
from pydantic import SecretStr

from litellm.llms.base_llm.auth.oauth_endpoint import (
    MAX_RESPONSE_BYTES,
    drop_reflected_credential,
    redact_oauth_error_body,
    validate_token_endpoint_url,
)
from litellm.llms.base_llm.auth.types import InsecureTokenUrl, TokenEndpointError

REFLECTED_MESSAGE: Final = "<redacted: response echoed the request>"


class TestRedactOauthErrorBody:
    def test_rfc6749_fields_are_kept_and_each_capped(self):
        body = {
            "error": "invalid_grant",
            "error_description": "d" * 300,
            "error_uri": "https://errors.example/e1",
        }
        result = redact_oauth_error_body(400, json.dumps(body))

        assert result == TokenEndpointError(status_code=400, redacted_body=result.redacted_body)
        assert "invalid_grant" in result.redacted_body
        assert "d" * 256 in result.redacted_body
        assert "d" * 257 not in result.redacted_body
        assert "https://errors.example/e1" in result.redacted_body

    def test_non_oauth_fields_are_never_rendered(self):
        body = {"detail": "internal id 4711", "error": "server_error"}
        result = redact_oauth_error_body(500, json.dumps(body))

        assert result.redacted_body == "error: server_error"

    def test_an_object_without_oauth_fields_gets_a_constant_message(self):
        result = redact_oauth_error_body(400, json.dumps({"detail": "internal id 4711"}))
        assert result.redacted_body == "error response carried no RFC 6749 fields"

    def test_nested_error_envelope_renders_readable_text(self):
        body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "federation_rule_id is not a well-formed fdrl_ tagged ID",
            },
        }
        result = redact_oauth_error_body(400, json.dumps(body))

        assert "invalid_request_error" in result.redacted_body
        assert "federation_rule_id is not a well-formed fdrl_ tagged ID" in result.redacted_body
        assert "{'" not in result.redacted_body

    def test_flat_rfc6749_shape_still_renders(self):
        body = {"error": "invalid_grant", "error_description": "bad request"}
        result = redact_oauth_error_body(400, json.dumps(body))

        assert result.redacted_body == "error: invalid_grant; error_description: bad request"

    def test_nested_error_message_is_capped_at_256_chars(self):
        body = {"error": {"type": "invalid_request_error", "message": "m" * 500}}
        result = redact_oauth_error_body(400, json.dumps(body))

        assert "m" * 256 in result.redacted_body
        assert "m" * 257 not in result.redacted_body

    def test_json_string_body_is_not_echoed(self):
        """A free-text body can carry back whatever was sent, so only structured OAuth fields are
        ever rendered into an error an operator or caller will see."""
        result = redact_oauth_error_body(400, json.dumps("s" * 500))
        assert result.redacted_body == "non-object error response omitted"
        assert "s" * 32 not in result.redacted_body

    def test_json_array_body_constant_message(self):
        result = redact_oauth_error_body(400, json.dumps(["a", "b"]))
        assert result.redacted_body == "non-object error response omitted"

    def test_plain_text_body_is_not_echoed(self):
        result = redact_oauth_error_body(502, "t" * 500)
        assert result.status_code == 502
        assert result.redacted_body == "non-JSON error response omitted"
        assert "t" * 32 not in result.redacted_body

    def test_oversized_body_is_never_parsed(self):
        body = '{"error": "' + "x" * MAX_RESPONSE_BYTES + '"}'
        result = redact_oauth_error_body(400, body)

        assert result.redacted_body == "oversized error response omitted"

    def test_sentinel_messages_pass_through_unchanged(self):
        result = redact_oauth_error_body(400, "oversized error response omitted")
        assert result.redacted_body == "oversized error response omitted"

    def test_reflected_assertion_is_dropped(self):
        """An endpoint that echoes the submitted assertion must not put it in the log or the error."""
        assertion = SecretStr("eyJhbGciOiJSUzI1NiJ9.REFLECTEDPAYLOAD.signature")
        body = {"error": "invalid_grant", "error_description": f"bad assertion {assertion.get_secret_value()}"}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert result.redacted_body == REFLECTED_MESSAGE
        assert "REFLECTEDPAYLOAD" not in result.redacted_body

    def test_assertion_reflected_from_an_offset_is_dropped(self):
        """Regression: the probe only looked at the assertion's first 24 characters, so an
        endpoint echoing it from any later offset shared no prefix and slipped through."""
        assertion = SecretStr("eyJhbGciOiJSUzI1NiJ9." + "A" * 40 + "PAYLOADMIDDLE" + "B" * 40 + ".signature")
        tail = assertion.get_secret_value()[24:]
        body = {"error": "invalid_grant", "error_description": tail}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert "PAYLOADMIDDLE" not in result.redacted_body
        assert tail[:40] not in result.redacted_body

    def test_a_secret_carrying_spaces_is_dropped_when_echoed_whole(self):
        """Regression on the redactor itself: comparing a compacted response against an
        uncompacted secret stopped matching hand-set passphrases, which are exactly the secrets
        most likely to be echoed and the ones an earlier contiguous match had caught."""
        assertion = SecretStr("correct horse battery staple, 42!")
        echoed = assertion.get_secret_value()
        body = {"error": "invalid_client", "error_description": f"secret {echoed} rejected"}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert echoed not in result.redacted_body

    def test_a_percent_encoded_secret_is_dropped(self):
        """A form-encoded grant puts the secret on the wire percent-escaped, so an echo of that
        shape has to be recognised without every caller enumerating it."""
        assertion = SecretStr("sUp3r+S3cret/Value=123")
        echoed = quote(assertion.get_secret_value(), safe="")
        body = {"error": "invalid_client", "error_description": f"rejected {echoed}"}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert echoed not in result.redacted_body

    def test_a_space_encoded_as_plus_is_dropped(self):
        """A form-encoded body writes a space as "+", not %20, so percent-decoding alone does not
        recover the secret and a passphrase echoed in its wire shape would travel on."""
        assertion = SecretStr("correct horse battery staple")
        echoed = urlencode({"client_secret": assertion.get_secret_value()}).split("=", 1)[1]
        body = {"error": "invalid_client", "error_description": f"rejected {echoed}"}

        assert "+" in echoed
        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert echoed not in result.redacted_body

    def test_several_wire_forms_are_all_compared(self):
        """The caller declares each shape it sent, since an encoding the redactor cannot reverse
        (base64 of id:secret) is only knowable there."""
        raw = SecretStr("sUp3rS3cretValue123")
        blob = SecretStr(base64.b64encode(b"litellm:sUp3rS3cretValue123").decode())
        body = {"error": "invalid_client", "error_description": f"bad {blob.get_secret_value()}"}

        result = redact_oauth_error_body(400, json.dumps(body), (raw, blob))

        assert blob.get_secret_value() not in result.redacted_body

    def test_a_fragment_shorter_than_a_long_run_is_dropped(self):
        """A slice too short to share a long contiguous run with the assertion is still assertion
        material, and repeated errors would hand it over piece by piece."""
        assertion = SecretStr("eyJhbGciOiJSUzI1NiJ9." + "A" * 60 + ".sigsigsig")
        fragment = assertion.get_secret_value()[30:48]
        body = {"error": "invalid_grant", "error_description": f"rejected near {fragment}"}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert fragment not in result.redacted_body

    def test_a_fragment_broken_up_by_delimiters_is_dropped(self):
        """Splitting the echo defeats a contiguous match, so the comparison ignores whatever the
        endpoint put between the pieces."""
        assertion = SecretStr("eyJhbGciOiJSUzI1NiJ9." + "A" * 60 + ".sigsigsig")
        piece = assertion.get_secret_value()[20:44]
        spaced = " ".join(piece[i : i + 6] for i in range(0, 24, 6))
        body = {"error": "invalid_grant", "error_description": spaced}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert spaced not in result.redacted_body

    def test_a_short_secret_is_still_matched_whole(self):
        """A Keycloak client secret can be shorter than the probe length; the whole value is
        compared in that case rather than a truncated prefix."""
        secret = SecretStr("short-secret")
        body = {"error": "invalid_client", "error_description": "rejected short-secret"}

        result = redact_oauth_error_body(400, json.dumps(body), secret)

        assert "short-secret" not in result.redacted_body

    def test_a_short_secret_echoed_in_its_wire_shape_is_dropped(self):
        """Regression: the run scan only ever compared eight-character windows, so a secret
        with fewer credential characters than that could never match once it came back
        percent-encoded rather than verbatim, and the whole-value check needs the raw form."""
        secret = SecretStr("p@ss w0rd!")
        echoed = quote(secret.get_secret_value(), safe="")
        body = {"error": "invalid_client", "error_description": f"rejected {echoed}"}

        assert secret.get_secret_value() not in echoed
        result = redact_oauth_error_body(400, json.dumps(body), secret)

        assert echoed not in result.redacted_body

    def test_an_unrelated_body_is_not_falsely_redacted(self):
        """The scan must not fire on a body that merely shares short runs with the assertion."""
        assertion = SecretStr("eyJhbGciOiJSUzI1NiJ9." + "Z" * 60 + ".signature")
        body = {"error": "invalid_grant", "error_description": "the federation rule was not found"}

        result = redact_oauth_error_body(400, json.dumps(body), assertion)

        assert "the federation rule was not found" in result.redacted_body


class TestDropReflectedCredential:
    def test_no_credential_passes_the_text_through(self):
        assert drop_reflected_credential("token_type was mac", None) == "token_type was mac"

    def test_an_empty_credential_never_matches(self):
        assert drop_reflected_credential("anything at all", SecretStr("")) == "anything at all"

    def test_a_credential_made_only_of_punctuation_never_matches(self):
        assert drop_reflected_credential("??? ###", SecretStr("!!!")) == "??? ###"

    def test_a_verbatim_echo_is_replaced_by_the_sentinel(self):
        secret = SecretStr("eyJhbGciOiJSUzI1NiJ9.PAYLOAD.signature")
        rendered = f"could not parse {secret.get_secret_value()}"

        assert drop_reflected_credential(rendered, secret) == REFLECTED_MESSAGE

    def test_a_percent_encoded_echo_is_replaced_by_the_sentinel(self):
        secret = SecretStr("sUp3r+S3cret/Value=123")
        rendered = f"rejected {quote(secret.get_secret_value(), safe='')}"

        assert drop_reflected_credential(rendered, secret) == REFLECTED_MESSAGE

    def test_unrelated_text_is_returned_unchanged(self):
        secret = SecretStr("eyJhbGciOiJSUzI1NiJ9." + "Z" * 60 + ".signature")

        assert drop_reflected_credential("token_type was mac", secret) == "token_type was mac"


class TestValidateTokenEndpointUrl:
    def test_https_is_returned_unchanged(self):
        assert validate_token_endpoint_url("https://token.example/v1") == "https://token.example/v1"

    def test_plain_http_is_rejected_naming_only_the_host(self):
        result = validate_token_endpoint_url("http://token.example/v1/oauth/token?client_secret=x")

        assert result == InsecureTokenUrl(host="token.example")
        assert "/v1/oauth/token" not in str(result)
        assert "client_secret" not in str(result)

    def test_a_url_without_a_scheme_is_rejected(self):
        assert validate_token_endpoint_url("token.example/v1") == InsecureTokenUrl(host="")

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080/v1/oauth/token",
            "http://127.0.0.1/v1/oauth/token",
            "http://[::1]/v1/oauth/token",
        ],
    )
    def test_localhost_http_is_allowed(self, url: str):
        assert validate_token_endpoint_url(url) == url

    def test_a_localhost_lookalike_is_still_rejected(self):
        assert validate_token_endpoint_url("http://localhost.example/v1") == InsecureTokenUrl(host="localhost.example")
