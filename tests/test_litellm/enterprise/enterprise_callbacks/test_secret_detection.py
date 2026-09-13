"""Tests for the hide-secrets guardrail (LIT-3548).

Covers the three defects from the ticket:
- ``apply_guardrail`` (the UI test playground path) must redact, not echo.
- Guardrail runs must record ``standard_logging_guardrail_information`` so
  Spend Logs / the guardrails monitor show activity, with hits ("mask" +
  masked_entity_count) distinguishable from clean requests ("allow").
- Defining ``apply_guardrail`` must NOT reroute proxied traffic off the
  native ``async_pre_call_hook`` (per-key opt-out and ``data["prompt"]``
  handling live only on the native path).
"""

import tempfile
import time

import pytest

from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
    _default_detect_secrets_config,
    _masked_entity_count,
)
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
OPENAI_KEY = "sk-test-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"
SHORT_OPENAI_KEY = "sk-12345"
UNICODE_DIGIT_SUFFIX = "sk-notification٣"
STRIPE_LIVE_KEY = f"sk_live_{'1234567890' * 3}"
URL_ENCODED_KEY = "Bearer%20sk-Ab3dEf6Gh7Ij8Kl9Mn0Pq2Rs3Tu4Vw5X"
AWS_KEYS = [f"AKIAIOSFODNN7EXAMPL{suffix}" for suffix in "FEDCBA"]


@pytest.fixture(autouse=True)
def _isolate_masked_entity_count():
    token = _masked_entity_count.set(None)
    yield
    _masked_entity_count.reset(token)


def _guardrail() -> _ENTERPRISE_SecretDetection:
    return _ENTERPRISE_SecretDetection(guardrail_name="hide-secrets", event_hook="pre_call", default_on=True)


def _recorded(request_data: dict) -> dict:
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    return entries[0]


def test_scan_message_preserves_benign_identifiers_and_xml_tags():
    guardrail = _guardrail()
    content = "<task-notification> model: claude-sonnet-4-5-20250929 </task-notification>"

    assert guardrail.scan_message_for_secrets(content) == []
    assert guardrail.redact_text(content) == content
    assert guardrail.redact_text("result = compute(x) </task-notification>") == (
        "result = compute(x) </task-notification>"
    )


def test_scan_message_preserves_quoted_benign_identifiers():
    guardrail = _guardrail()
    content = '{"content-type": "application/json", "model": "claude-sonnet-4-5-20250929"}'

    assert guardrail.scan_message_for_secrets(content) == []
    assert guardrail.redact_text(content) == content


@pytest.mark.parametrize(
    "content,secret",
    [
        ("REDIS_PASSWORD=aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("SESSION_SECRET=Kp7Nq2Wz9Bt4Xr6Vm1Ls", "Kp7Nq2Wz9Bt4Xr6Vm1Ls"),
        ('{"db_password": "Tq8Zm2XpLv9KdNbRcYw3"}', "Tq8Zm2XpLv9KdNbRcYw3"),
        ("api_secret: Zx4Kp9Lm2Qr7Ns3Vt", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("password = hunter2brahms9x", "hunter2brahms9x"),
        ("client_secret=Hq7Zm3XkLp9Wd2Nb", "Hq7Zm3XkLp9Wd2Nb"),
        ('apiKey: "aB3dE6gH9jK2mN5p"', "aB3dE6gH9jK2mN5p"),
        ('{"clientSecret": "Kp7Nq2Wz9Bt4Xr6Vm1Ls"}', "Kp7Nq2Wz9Bt4Xr6Vm1Ls"),
        ('dbPassword = "Zx4Kp9Lm2Qr7Ns3Vt"', "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("MY_APP_DB_PASSWORD=Kp7Nq2Wz9Bt4Xr6Vm1Ls", "Kp7Nq2Wz9Bt4Xr6Vm1Ls"),
        ("x_api_key: 8f3Kd9Lm2Qr7Ns3Vt", "8f3Kd9Lm2Qr7Ns3Vt"),
        ("password: Zm9vYmFyYmF6+abc/def123=", "Zm9vYmFyYmF6+abc/def123="),
        ("REDIS_PASSWORD=correcthorsebattery", "correcthorsebattery"),
        ('SECRET_KEY = "django-insecure-9v2xk4qw8z"', "django-insecure-9v2xk4qw8z"),
        (
            "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ),
        ("password=aB3dE6gH9jK2", "aB3dE6gH9jK2"),
        ("api_key: hunter2!brahms", "hunter2!brahms"),
        ('db_password: "p@ssw0rd!2026"', "p@ssw0rd!2026"),
        (
            'url: "postgresql://user:s3cr3t@db-host:5432/app"',
            "postgresql://user:s3cr3t@db-host:5432/app",
        ),
        (
            'db_password: "postgresql://user:s3cr3t@db-host:5432/app"',
            "postgresql://user:s3cr3t@db-host:5432/app",
        ),
        (
            'signing_secret_url: "https://example.com/cb?sig=Zx4Kp9Lm2Qr7Ns3Vt"',
            "https://example.com/cb?sig=Zx4Kp9Lm2Qr7Ns3Vt",
        ),
        (
            'redis_secret_url: "redis://:Zx4Kp9Lm2Qr7Ns3Vt@cache-host:6379/0"',
            "Zx4Kp9Lm2Qr7Ns3Vt",
        ),
        ("password=2026-09-08T17:38:40Zbrahms", "2026-09-08T17:38:40Zbrahms"),
        (
            '{"password": "YOUR_API_KEY_HERE", "client_secret": "correcthorsebattery"}',
            "correcthorsebattery",
        ),
        ("docker run -e REDIS_PASSWORD=aB3dE6gH9jK2mN5p \\\n  -e REDIS_PORT=6379 redis", "aB3dE6gH9jK2mN5p"),
        ("DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt && echo done", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("password = Zx4Kp9Lm2Qr7Ns3Vt  # rotate me", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("my db password: Zx4Kp9Lm2Qr7Ns3Vt.", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("export DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt DB_HOST=db.internal", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt; systemctl restart app", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt | tee creds.txt", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt > setup.log", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("docker run -e DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt --name app postgres", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("password=correcthorsebattery please", "correcthorsebattery"),
        ("DB_PASSWORD = Zx4Kp9Lm2Qr7Ns3Vt; systemctl restart app", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD = Zx4Kp9Lm2Qr7Ns3Vt \\", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD = Zx4Kp9Lm2Qr7Ns3Vt DB_HOST=db.internal", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD = Zx4Kp9Lm2Qr7Ns3Vt --db-host=db.internal", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("DB_PASSWORD = Zx4Kp9Lm2Qr7Ns3Vt DEBUG=", "Zx4Kp9Lm2Qr7Ns3Vt"),
    ],
    ids=[
        "env-password",
        "env-secret",
        "json-field",
        "yaml-field",
        "bare-assignment",
        "client-secret",
        "camel-case-key",
        "camel-case-secret",
        "camel-case-password",
        "namespaced-env",
        "underscored-header",
        "base64-padding",
        "digit-free-value",
        "django-secret-key",
        "slashed-aws-secret",
        "shortest-accepted-value",
        "punctuation-bearing-password",
        "symbol-heavy-password",
        "connection-string-under-a-url-key",
        "connection-string-under-a-credential-key",
        "signed-url-under-a-credential-key",
        "password-only-url-under-a-credential-key",
        "timestamp-prefixed-password",
        "credential-after-a-rejected-placeholder",
        "docker-flag-with-a-line-continuation",
        "shell-command-after-the-value",
        "inline-comment-after-the-value",
        "sentence-ending-in-the-value",
        "second-assignment-after-the-value",
        "semicolon-after-the-value",
        "pipe-after-the-value",
        "redirect-after-the-value",
        "docker-flag-after-the-value",
        "prose-after-a-shell-assignment",
        "spaced-assignment-then-a-shell-command",
        "spaced-assignment-then-a-line-continuation",
        "spaced-assignment-then-a-second-assignment",
        "spaced-assignment-then-a-dashed-flag",
        "spaced-assignment-then-an-empty-assignment",
    ],
)
def test_scan_message_redacts_credentials_assigned_to_credential_keys(content, secret):
    guardrail = _guardrail()

    assert secret not in guardrail.redact_text(content)


def test_scan_message_redacts_only_the_first_token_of_a_shell_assignment():
    guardrail = _guardrail()
    content = "docker run -e REDIS_PASSWORD=aB3dE6gH9jK2mN5p \\\n  -e REDIS_PORT=6379 redis && echo done"

    assert (
        guardrail.redact_text(content)
        == "docker run -e REDIS_PASSWORD=[REDACTED] \\\n  -e REDIS_PORT=6379 redis && echo done"
    )


@pytest.mark.parametrize("operator", [";", "&&", "|"])
def test_scan_message_keeps_a_shell_operator_glued_to_the_value(operator):
    guardrail = _guardrail()

    assert (
        guardrail.redact_text(f"DB_PASSWORD=Zx4Kp9Lm2Qr7Ns3Vt{operator} systemctl restart app")
        == f"DB_PASSWORD=[REDACTED]{operator} systemctl restart app"
    )


def test_scan_message_closes_a_yaml_block_at_the_next_unindented_line():
    guardrail = _guardrail()
    content = "api_key: >\n  aB3dE6gH9jK2mN5p\nSteps\n  Rotate-Before-Friday please"

    assert guardrail.redact_text(content) == "api_key: >\n  [REDACTED]\nSteps\n  Rotate-Before-Friday please"


def test_scan_message_redacts_every_credential_on_one_line():
    guardrail = _guardrail()
    content = '{"db_password": "Tq8Zm2XpLv9KdNbRcYw3", "client_secret": "correcthorsebattery"}'

    assert guardrail.redact_text(content) == '{"db_password": "[REDACTED]", "client_secret": "[REDACTED]"}'


@pytest.mark.parametrize(
    "content",
    [
        "The user forgot their password and asked for a reset link",
        "Rotate the client secret every 90 days",
        "The secret: keep it quiet",
        "My password: correct horse battery staple",
        "secretary: Maria Gonzalez",
        "password_reset_email: Please click the link below to reset",
        'config = {"api_key": "YOUR_API_KEY_HERE"}',
        "api_key: <your-key-here>",
        '{"max_tokens": 4096, "model": "gpt-4o-mini"}',
        'def get_api_key():\n    return os.environ["OPENAI_API_KEY"]',
        '    valid_token = UserAPIKeyAuth(user_id="u1")',
        'password = get_password(user, "prod")',
        "monkey=aB3dE6gH9jK2mN5p",
        "idempotency_key: req_2026090712000000",
        'cache_key = "u1_user_api_key_user_id"',
        "the key: 2026-09-07T12:00:00Z",
        "api_key: os.environ/E2B_API_KEY",
        "langfuse_secret: os.environ/LANGFUSE_PROJECT1_SECRET",
        "api_key = OPENAI_API_KEY",
        "password = pwd12345678",
        "model_key: gpt-4o-mini-2024-07-18",
        "openrouter/anthropic/claude-3-5-sonnet-20240620",
        '{"content-type": "application/json"}',
        "passwordless_login: enabled-for-all-users",
        'password: "I forgot mine, can you reset it"',
        "secret_sauce: tomatoes-basil-garlic-oregano",
        "user_secret_question: what-was-your-first-pet",
        "password_reset_url: example.com/reset-password/flow",
        "private_key_path: keys/prod/server-cert.pem",
        "litellm.completion(model=model, api_key=openai_api_key)",
        "params['aws_secret_access_key'] = aws_secret_access_key",
        'api_key = "OPENAI_API_KEY"',
        "model_list:\n  - litellm_params:\n      api_key: 'PERPLEXITY_API_KEY'",
        'config = build(_provider("ve_missing", api_key_env="VE_MISSING_KEY"))',
        "api_key = get_api_key_from_env()",
        "api_key = get_secret_str(MISTRAL_OCR_API_KEY_ENV_VAR)",
        "secret_manager = MagicMock(spec=BaseSecretManager)",
        "api_key = self.resolve_server_api_key(",
        "api_key = sys.argv[1]",
        "password = credentials[environment]",
        'api_key_created_at: "2026-09-08T17:38:40Z"',
        'api_key_expires_at: "2026-09-08T17:38:40.123456+05:30"',
        'password_reset_url: "https://example.com/reset-password/flow"',
        'secret_docs_url: "https://example.com/reset-password/flow#step-2"',
        '{"api_key_created_at": "2026-09-08T17:38:40Z", "password_reset_url": "https://example.com/reset/flow"}',
        "secret_sauce: tomatoes-basil-garlic-oregano.",
        "secret_docs_url: https://example.com/docs/keys, then rotate",
        "api_key_created_at: 2026-09-08T17:38:40Z; api_key_env: OPENAI_API_KEY!",
        "api_key: $OPENAI_API_KEY",
        'api_key: "${OPENAI_API_KEY}"',
        "private_key_path: /keys/prod/server-cert.pem",
        "password_hint: your usual one followed by Ticket-LIT7049-Suffix",
        "Translate this recipe note into French:\nsecret_sauce: Worcestershire sauce",
        "api_key = Massachusetts (the state, not a key)",
        "secret_sauce:Worcestershire sauce",
        "password: correctHorseBattery != anotherValue",
    ],
    ids=[
        "prose-password",
        "prose-secret",
        "colon-prose-secret",
        "colon-prose-password",
        "secretary",
        "sentence-after-keyword",
        "uppercase-placeholder",
        "templated-placeholder",
        "max-tokens",
        "code-paste",
        "constructor-call",
        "indirect-reference",
        "word-ending-in-key",
        "idempotency-key",
        "cache-key",
        "timestamp-after-key",
        "env-reference",
        "env-reference-nested",
        "env-variable-name",
        "below-minimum-length",
        "model-name",
        "namespaced-model-name",
        "media-type",
        "hyphenated-english",
        "quoted-sentence-under-a-credential-key",
        "hyphenated-phrase",
        "hyphenated-question",
        "url-under-credential-key",
        "path-under-credential-key",
        "snake-case-argument",
        "snake-case-assignment",
        "quoted-env-variable-name",
        "quoted-env-name-in-a-config",
        "quoted-env-name-in-a-code-paste",
        "bare-call",
        "call-with-an-argument",
        "keyword-argument-call",
        "unclosed-call",
        "positional-subscript",
        "keyed-subscript",
        "timestamp-under-a-credential-key",
        "offset-timestamp-under-a-credential-key",
        "url-under-a-credential-key",
        "fragment-url-under-a-credential-key",
        "metadata-object-under-credential-keys",
        "hyphenated-english-ending-a-sentence",
        "url-followed-by-a-clause",
        "timestamp-and-env-name-with-trailing-punctuation",
        "shell-variable-reference",
        "quoted-braced-shell-variable-reference",
        "absolute-path-under-a-credential-key",
        "sentence-holding-a-later-mixed-case-token",
        "capitalized-word-starting-a-phrase",
        "capitalized-word-before-a-parenthetical",
        "yaml-scalar-without-a-space-after-the-colon",
        "comparison-operator-after-the-value",
    ],
)
def test_scan_message_keeps_benign_values(content):
    guardrail = _guardrail()

    assert guardrail.scan_message_for_secrets(content) == []
    assert guardrail.redact_text(content) == content


@pytest.mark.parametrize(
    "value,redacted",
    [("aB3dE6gH9jK2", True), ("aB3dE6gH9jK", False)],
    ids=["at-minimum-length", "below-minimum-length"],
)
def test_credential_keyword_detector_honours_its_minimum_length(value, redacted):
    guardrail = _guardrail()

    assert (value not in guardrail.redact_text(f"password={value}")) is redacted


@pytest.mark.parametrize(
    "value,redacted",
    [("aB3dE6gH9jK2", True), ("aB3dE6gH9jK", False)],
    ids=["at-default-minimum-length", "below-default-minimum-length"],
)
def test_credential_keyword_detector_defaults_its_minimum_length(value, redacted):
    guardrail = _ENTERPRISE_SecretDetection(
        guardrail_name="hide-secrets",
        event_hook="pre_call",
        default_on=True,
        detect_secrets_config={
            "plugins_used": [
                {key: setting for key, setting in plugin.items() if key != "minimum_length"}
                for plugin in _default_detect_secrets_config["plugins_used"]
            ]
        },
    )

    assert (value not in guardrail.redact_text(f"password={value}")) is redacted


def test_credential_keyword_detector_honours_keyword_exclude():
    guardrail = _ENTERPRISE_SecretDetection(
        guardrail_name="hide-secrets",
        event_hook="pre_call",
        default_on=True,
        detect_secrets_config={
            "plugins_used": [
                {**plugin, "keyword_exclude": "fixture_"} if plugin["name"] == "CredentialKeywordDetector" else plugin
                for plugin in _default_detect_secrets_config["plugins_used"]
            ]
        },
    )
    content = "fixture_password=aB3dE6gH9jK2mN5p\npassword=Kp7Nq2Wz9Bt4Xr6Vm1Ls"

    assert guardrail.redact_text(content) == "fixture_password=aB3dE6gH9jK2mN5p\npassword=[REDACTED]"


@pytest.mark.parametrize("minimum_length", ["12", 0, -1, 1.5], ids=["string", "zero", "negative", "float"])
def test_credential_keyword_detector_rejects_an_unusable_minimum_length(minimum_length):
    guardrail = _ENTERPRISE_SecretDetection(
        guardrail_name="hide-secrets",
        event_hook="pre_call",
        default_on=True,
        detect_secrets_config={
            "plugins_used": [
                {**plugin, "minimum_length": minimum_length}
                if plugin["name"] == "CredentialKeywordDetector"
                else plugin
                for plugin in _default_detect_secrets_config["plugins_used"]
            ]
        },
    )

    with pytest.raises(ValueError, match="minimum_length"):
        guardrail.scan_message_for_secrets("password=aB3dE6gH9jK2mN5p")


@pytest.mark.parametrize(
    "content",
    [
        "[db\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "[\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "[note] have a look\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "]\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "[]\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
    ],
    ids=["unclosed", "bare-bracket", "bracketed-prose", "stray-close", "empty-header"],
)
def test_scan_message_reads_a_config_with_a_broken_section_header(content):
    guardrail = _guardrail()

    assert "Zx4Kp9Lm2Qr7Ns3Vt" not in guardrail.redact_text(content)


@pytest.mark.parametrize(
    "content",
    [
        "=orphan\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "  indented before any key\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "greeting = %(name)s\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
        "token = a\x00b\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n",
    ],
    ids=["empty-key", "leading-continuation", "interpolation", "nul-byte"],
)
def test_scan_message_reads_lines_that_a_stock_ini_parser_rejects(content):
    guardrail = _guardrail()

    assert "Zx4Kp9Lm2Qr7Ns3Vt" not in guardrail.redact_text(content)


def test_scan_message_reads_a_config_that_repeats_a_section():
    guardrail = _guardrail()
    content = "[db]\nhost = localhost\n[db]\npassword = Zx4Kp9Lm2Qr7Ns3Vt\n"

    assert "Zx4Kp9Lm2Qr7Ns3Vt" not in guardrail.redact_text(content)


def test_scan_message_keeps_every_value_when_a_config_repeats_a_key():
    guardrail = _guardrail()
    content = (
        "model_list:\n"
        "  - model_name: gpt-4o\n    litellm_params:\n      api_key: aB3dE6gH9jK2mN5p\n"
        "  - model_name: claude\n    litellm_params:\n      api_key: Kp7Nq2Wz9Bt4Xr6Vm1Ls\n"
    )

    redacted = guardrail.redact_text(content)

    assert "aB3dE6gH9jK2mN5p" not in redacted
    assert "Kp7Nq2Wz9Bt4Xr6Vm1Ls" not in redacted


@pytest.mark.parametrize(
    "content,secret",
    [
        (
            f"api_key: {OPENAI_KEY}\nREDIS_PASSWORD=aB3dE6gH9jK2mN5p",
            "aB3dE6gH9jK2mN5p",
        ),
        (
            f"OPENAI_API_KEY={OPENAI_KEY}\nDB_PASSWORD=Kp7Nq2Wz9Bt4Xr6Vm1Ls",
            "Kp7Nq2Wz9Bt4Xr6Vm1Ls",
        ),
        (
            f"api_key: {OPENAI_KEY}\npassword =\n    Zx4Kp9Lm2Qr7Ns3Vt",
            "Zx4Kp9Lm2Qr7Ns3Vt",
        ),
        (
            "Here is my config, can you review it?\nREDIS_PASSWORD=aB3dE6gH9jK2mN5p",
            "aB3dE6gH9jK2mN5p",
        ),
        (
            "REDIS_PASSWORD=aB3dE6gH9jK2mN5p\nCan you tell me what is wrong with it?",
            "aB3dE6gH9jK2mN5p",
        ),
        (
            "Hi team\nplease rotate this before Friday\ndb_password=Zx4Kp9Lm2Qr7Ns3Vt\nthanks!",
            "Zx4Kp9Lm2Qr7Ns3Vt",
        ),
        (
            "model_list:\n  - model_name: gpt-4o\n    litellm_params:\n      api_key: aB3dE6gH9jK2mN5p\n",
            "aB3dE6gH9jK2mN5p",
        ),
        ("api_key: >\n  aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("api_key: |-\n  aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("secret= \\\n    aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("password =\n# rotate me\n    Zx4Kp9Lm2Qr7Ns3Vt", "Zx4Kp9Lm2Qr7Ns3Vt"),
        ("api_key =\n; rotate me\n    aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("  # pasted from the vault\napi_key=aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("  [db]\napi_key=aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
        ("    pasted with a leading indent\napi_key=aB3dE6gH9jK2mN5p", "aB3dE6gH9jK2mN5p"),
    ],
    ids=[
        "flat-assignment",
        "env-file",
        "continuation-line",
        "prose-before",
        "prose-after",
        "prose-both-sides",
        "indented-config",
        "yaml-folded-block",
        "yaml-literal-block",
        "backslash-continuation",
        "comment-inside-a-value",
        "semicolon-comment-inside-a-value",
        "indented-comment-above",
        "indented-section-header-above",
        "indented-prose-above",
    ],
)
def test_scan_message_still_sees_assignments_sharing_a_message_with_a_vendor_key(content, secret):
    guardrail = _guardrail()

    redacted = guardrail.redact_text(content)
    assert secret not in redacted
    assert OPENAI_KEY not in redacted


def test_environment_reference_filter_only_drops_the_whole_value():
    guardrail = _guardrail()

    for reference in ("os.environ/OPENAI_API_KEY", "os.environ/e2b_api_key"):
        assert guardrail.redact_text(f"password={reference}") == f"password={reference}"
    assert guardrail.redact_text("password=notos.environ/OPENAI_API_KEY") == ("password=[REDACTED]")


def test_environment_variable_names_are_dropped_only_for_the_keyword_plugin():
    guardrail = _guardrail()

    assert guardrail.redact_text("password=REDIS_PASSWORD") == "password=REDIS_PASSWORD"
    assert guardrail.scan_message_for_secrets('k = "ABCD1234_EFGH5678_IJKLMN"') == [
        {"type": "Base64 High Entropy String", "value": "ABCD1234_EFGH5678_IJKLMN"}
    ]


def test_masked_entity_count_keeps_the_vendor_type_beside_the_entropy_type():
    guardrail = _guardrail()
    _masked_entity_count.set({})

    guardrail.redact_text('k = "ghp_abcdefghijklmnopqrstuvwxyzABCDEF1234"')

    assert _masked_entity_count.get() == {
        "Base64 High Entropy String": 1,
        "GitHub Token": 1,
    }


@pytest.mark.parametrize(
    "content",
    [
        f"api_key: '{OPENAI_KEY}'\n"
        + "a: &a ["
        + ", ".join(['"x"'] * 9)
        + "]\n"
        + "".join(f"{chr(98 + i)}: &{chr(98 + i)} [" + ", ".join([f"*{chr(97 + i)}"] * 9) + "]\n" for i in range(7)),
        f"api_key: '{OPENAI_KEY}'\ndeep: " + "[" * 400 + "]" * 400,
        f"api_key: '{OPENAI_KEY}'\nbroken: [unclosed",
    ],
    ids=["anchor-expansion", "deep-nesting", "unparseable"],
)
def test_scan_message_contains_hostile_config_text(content, monkeypatch, tmp_path):
    guardrail = _guardrail()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)

    started = time.perf_counter()
    found = guardrail.scan_message_for_secrets(content)

    assert time.perf_counter() - started < 10.0
    assert OPENAI_KEY in [secret["value"] for secret in found]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "content",
    [
        f"api_key = '{OPENAI_KEY}'\nbase = abcdefghijkl\npassword = x\n    %(base)sZZZZQQQQ\n",
        "base = abcdefghijkl\npassword = x\n    %(base)sZZZZQQQQ\n",
        f"api_key = '{OPENAI_KEY}'\nbase = Kp7Nq2Wz9Bt4\npassword = x\n"
        "    %(base)s-primary\nnote = Kp7Nq2Wz9Bt4-primary is the hostname\n",
        'base = "abcdefghijkl"\npassword = "%(base)sZZZZQQQQ"\n',
    ],
    ids=[
        "vendor-key-present",
        "no-vendor-key",
        "value-echoed-elsewhere",
        "quoted-interpolation",
    ],
)
def test_scan_message_never_reports_a_value_the_message_does_not_hold(content):
    guardrail = _guardrail()

    for secret in guardrail.scan_message_for_secrets(content):
        assert secret["value"] in content


def test_scan_message_leaves_unrelated_text_alone_when_a_value_is_echoed():
    guardrail = _guardrail()
    content = (
        f"api_key = '{OPENAI_KEY}'\nbase = Kp7Nq2Wz9Bt4\npassword = x\n"
        "    %(base)s-primary\nnote = Kp7Nq2Wz9Bt4-primary is the hostname\n"
    )

    assert "note = Kp7Nq2Wz9Bt4-primary is the hostname" in guardrail.redact_text(content)


def test_masked_entity_count_counts_each_secret_once():
    guardrail = _guardrail()
    _masked_entity_count.set({})

    guardrail.redact_text(f"first {OPENAI_KEY} second {OPENAI_KEY}")

    assert _masked_entity_count.get() == {"Strict OpenAI API Key": 1}


def test_scan_message_redacts_every_openai_key_occurrence():
    guardrail = _guardrail()
    content = f"first {OPENAI_KEY}, second {OPENAI_KEY}"

    assert guardrail.redact_text(content) == "first [REDACTED], second [REDACTED]"


def test_scan_message_redacts_short_numeric_openai_like_values():
    guardrail = _guardrail()

    assert guardrail.redact_text(f"value {SHORT_OPENAI_KEY}") == "value [REDACTED]"


def test_scan_message_requires_ascii_digits_for_openai_like_values():
    guardrail = _guardrail()

    assert guardrail.scan_message_for_secrets(UNICODE_DIGIT_SUFFIX) == []
    assert guardrail.redact_text(UNICODE_DIGIT_SUFFIX) == UNICODE_DIGIT_SUFFIX


def test_scan_message_redacts_openai_key_after_separator():
    guardrail = _guardrail()

    assert guardrail.redact_text(f"openai_{OPENAI_KEY} key-{OPENAI_KEY}") == ("openai_[REDACTED] key-[REDACTED]")
    assert guardrail.redact_text(URL_ENCODED_KEY) == "Bearer%20[REDACTED]"


def test_scan_message_does_not_stop_openai_key_at_token_characters():
    guardrail = _guardrail()

    assert guardrail.redact_text("key sk-proj-abcde12345/extra") == "key [REDACTED]/extra"


def test_scan_message_stays_linear_on_repeated_sk_separators():
    guardrail = _guardrail()
    content = "-sk-" * 25_000

    started = time.perf_counter()
    assert guardrail.scan_message_for_secrets(content) == []
    assert time.perf_counter() - started < 2.0


@pytest.mark.parametrize(
    "content",
    [
        f"api_key: '{OPENAI_KEY}'\npassword=" + "a-" * 10_000 + "!",
        f"api_key: '{OPENAI_KEY}'\npassword:" + '"' * 20_000,
        f"api_key: '{OPENAI_KEY}'\n" + "api_key:" * 10_000,
        f"api_key: '{OPENAI_KEY}'\nsecret=" + "aB3dE6gH9jK2mN5p " * 2_000,
        f"api_key: '{OPENAI_KEY}'\n" + "\n".join(f"password{i}=aB3dE6gH9jK2mN5p{i}" for i in range(3_000)),
    ],
    ids=[
        "value-run",
        "quote-run",
        "keyword-run",
        "value-repeat",
        "assignment-flood",
    ],
)
def test_scan_message_stays_linear_on_adversarial_credential_lines(content):
    guardrail = _guardrail()

    started = time.perf_counter()
    guardrail.redact_text(content)
    assert time.perf_counter() - started < 10.0


def test_scan_message_redacts_whole_stripe_live_key():
    guardrail = _guardrail()

    assert guardrail.redact_text(f"stripe {STRIPE_LIVE_KEY} end") == "stripe [REDACTED] end"


def test_scan_message_returns_matches_in_stable_order():
    guardrail = _guardrail()
    detected = guardrail.scan_message_for_secrets(" ".join(AWS_KEYS))

    assert [secret["value"] for secret in detected] == sorted(AWS_KEYS)


def test_scan_message_replaces_longest_overlapping_match_first():
    guardrail = _guardrail()
    content = f'token = "{OPENAI_KEY}/extra"'

    values = [secret["value"] for secret in guardrail.scan_message_for_secrets(content)]
    assert values == [f"{OPENAI_KEY}/extra", OPENAI_KEY]
    assert guardrail.redact_text(content) == 'token = "[REDACTED]"'


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_secrets():
    """Playground path: the returned texts must carry [REDACTED], not the secret."""
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": [f"my key is {AWS_KEY}, keep it safe"]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"] == ["my key is [REDACTED], keep it safe"]

    recorded = _recorded(request_data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "mask"
    assert recorded["guardrail_provider"] == "hide-secrets"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_apply_guardrail_clean_text_records_allow():
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["nothing sensitive here"]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"] == ["nothing sensitive here"]

    recorded = _recorded(request_data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "allow"
    assert recorded["masked_entity_count"] == {}


@pytest.mark.asyncio
async def test_pre_call_hook_records_mask_with_entity_count():
    """Live-traffic path: a redaction must be visible in spend-log telemetry."""
    guardrail = _guardrail()
    data = {
        "messages": [{"role": "user", "content": f"use {AWS_KEY} for auth"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == "use [REDACTED] for auth"

    recorded = _recorded(data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "mask"
    assert recorded["guardrail_provider"] == "hide-secrets"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_pre_call_hook_clean_request_records_allow():
    """A request with no secrets must be distinguishable from a redacted one."""
    guardrail = _guardrail()
    data = {
        "messages": [{"role": "user", "content": "what's the weather"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    recorded = _recorded(data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "allow"
    assert recorded["masked_entity_count"] == {}


@pytest.mark.asyncio
async def test_pre_call_hook_opt_out_records_nothing():
    """A key with permissions={"hide_secrets": False} skips redaction, so no
    telemetry is recorded: every reader of a recorded entry (guardrail usage
    tracking, compliance checks, the spend-log viewer) counts it as a run."""
    guardrail = _guardrail()
    content = f"my key is {AWS_KEY}"
    data = {"messages": [{"role": "user", "content": content}], "metadata": {}}

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(permissions={"hide_secrets": False}),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == content  # untouched
    assert "standard_logging_guardrail_information" not in data["metadata"]


@pytest.mark.asyncio
async def test_pre_call_hook_still_redacts_text_completion_prompt():
    """data["prompt"] (str and list) is a native-hook-only surface; it must
    keep redacting now that the class also implements apply_guardrail."""
    guardrail = _guardrail()
    data = {"prompt": f"key {AWS_KEY} end", "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["prompt"] == "key [REDACTED] end"

    guardrail = _guardrail()
    data = {"prompt": [f"key {AWS_KEY}", "clean"], "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["prompt"] == ["key [REDACTED]", "clean"]


def test_proxied_traffic_stays_on_native_hooks():
    """Implementing apply_guardrail must not reroute proxied requests onto the
    unified path: that path skips ``should_run_check`` (per-key opt-out) and
    never sees ``data["prompt"]``."""
    guardrail = _guardrail()
    assert guardrail.uses_apply_guardrail_interface() is True
    assert guardrail._deployment_hook_target() is guardrail


@pytest.mark.asyncio
async def test_apply_guardrail_without_texts_records_nothing():
    """No inputs means nothing was inspected, so no "allow" row is recorded.
    Empty strings count as no input: there is no content to inspect."""
    guardrail = _guardrail()

    empty_variants: list[list[str]] = [[], ["", ""]]
    for texts in empty_variants:
        request_data: dict = {"metadata": {}}
        result = await guardrail.apply_guardrail(
            inputs={"texts": texts}, request_data=request_data, input_type="request"
        )
        assert result == {"texts": texts}
        assert "standard_logging_guardrail_information" not in request_data["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
                    }
                ],
                "metadata": {},
            },
            id="image_only",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": ""}], "metadata": {}},
            id="empty_message",
        ),
        pytest.param({"prompt": "", "metadata": {}}, id="empty_prompt"),
        pytest.param({"prompt": ["", ""], "metadata": {}}, id="empty_prompt_list"),
    ],
)
async def test_pre_call_hook_without_inspectable_text_records_nothing(data: dict):
    """A payload the guardrail could not inspect (image-only content, empty
    strings) must not record an "allow" run: monitoring would count a check
    that never looked at any text."""
    guardrail = _guardrail()

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert "standard_logging_guardrail_information" not in data["metadata"]


@pytest.mark.asyncio
async def test_pre_call_hook_mixed_prompt_list_still_redacts_and_records():
    """A prompt list mixing empty and real strings is inspected, so the run is
    recorded and the non-empty entry is still redacted."""
    guardrail = _guardrail()
    data = {"prompt": ["", f"key {AWS_KEY}"], "metadata": {}}

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["prompt"] == ["", "key [REDACTED]"]
    recorded = _recorded(data)
    assert recorded["guardrail_response"] == "mask"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_legacy_nameless_instance_records_nothing():
    """``litellm_settings.callbacks: ["hide_secrets"]`` builds an arg-less
    instance with no guardrail_name. It still redacts, but recording a nameless
    entry would flip every spend row's guardrail status with nothing to join on."""
    guardrail = _ENTERPRISE_SecretDetection()
    data = {
        "messages": [{"role": "user", "content": f"use {AWS_KEY} for auth"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == "use [REDACTED] for auth"
    assert "standard_logging_guardrail_information" not in data["metadata"]
