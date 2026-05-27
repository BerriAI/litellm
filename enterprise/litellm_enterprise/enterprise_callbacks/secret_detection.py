# +-------------------------------------------------------------+
#
#           Use SecretDetection /moderations for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import re
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import tempfile
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import walk_user_text

GUARDRAIL_NAME = "hide_secrets"

# Matches a full PEM private-key block (header + base64 body + footer) across
# multiple lines.  detect-secrets is line-based and only records the BEGIN
# header as the secret value; ``_expand_private_key_values`` uses this
# pattern to promote that header-only value to the full block so every
# str.replace call site redacts the entire key, not just the first line.
# Line-anchored PEM markers.  Used by ``_find_pem_blocks`` to walk the
# input one line at a time, which is O(n) in the input length and avoids
# any regex backtracking on adversarial inputs (e.g. many BEGIN headers
# with no matching footer).
_PEM_BEGIN_LINE_RE = re.compile(r"^-----BEGIN[ A-Z]*PRIVATE KEY-----\s*$")
_PEM_END_LINE_RE = re.compile(r"^-----END[ A-Z]*PRIVATE KEY-----\s*$")
# Per-block scan cap: PEM keys are at most a few KB; this bounds how many
# lines we'll walk forward looking for the matching END footer before
# giving up and treating the BEGIN as an orphan (truncated) block.
_PEM_BLOCK_MAX_LINES = 256


def _find_pem_blocks(text: str) -> List[str]:
    """Return the full text of every PEM private-key block in *text*.

    Walks line-by-line in O(n).  For each ``-----BEGIN ... PRIVATE KEY-----``
    line we look forward up to ``_PEM_BLOCK_MAX_LINES`` for a matching
    ``-----END ... PRIVATE KEY-----``.  If an END is found, the full block
    (BEGIN through END, including original newlines) is returned.  If we
    hit another BEGIN, the end of input, or the line cap first, we return
    everything from the BEGIN through the line just before whatever made
    us stop, so a truncated PEM body (no END footer) is still treated as
    secret material and redacted.

    Returns a list of block strings in order of appearance.  No state is
    shared across blocks.
    """
    blocks: List[str] = []
    # Use ``splitlines(keepends=True)`` so reconstructing a block preserves
    # the original line terminators.
    lines = text.splitlines(keepends=True)
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].rstrip("\r\n")
        if _PEM_BEGIN_LINE_RE.match(stripped):
            end_idx = None
            for j in range(i + 1, min(n, i + 1 + _PEM_BLOCK_MAX_LINES)):
                inner = lines[j].rstrip("\r\n")
                if _PEM_END_LINE_RE.match(inner):
                    end_idx = j
                    break
                if _PEM_BEGIN_LINE_RE.match(inner):
                    # Another BEGIN before a matching END -> treat the
                    # current block as orphaned, stop just before this one.
                    break
            if end_idx is not None:
                # Strip trailing newline from the END line for a cleaner
                # value (matches the historical regex behavior).
                block = "".join(lines[i:end_idx + 1]).rstrip("\r\n")
                blocks.append(block)
                i = end_idx + 1
                continue
            else:
                # Orphan BEGIN: redact from the BEGIN line through whatever
                # we walked, which is at most _PEM_BLOCK_MAX_LINES.  This
                # closes the "BEGIN + body, no END" partial-redaction gap.
                stop = min(n, i + 1 + _PEM_BLOCK_MAX_LINES)
                # If we stopped because of a next BEGIN, ``j`` is where we
                # stopped; otherwise stop at the cap or end of input.
                # We use the loop variable ``j``, but it may not be defined
                # if there were no lines after BEGIN; recompute safely.
                if i + 1 >= n:
                    block = lines[i].rstrip("\r\n")
                else:
                    last = stop
                    for j in range(i + 1, stop):
                        inner = lines[j].rstrip("\r\n")
                        if _PEM_BEGIN_LINE_RE.match(inner):
                            last = j
                            break
                    block = "".join(lines[i:last]).rstrip("\r\n")
                blocks.append(block)
                i = (last if i + 1 < n else i + 1)
                continue
        i += 1
    return blocks



def _expand_private_key_values(
    detected_secrets: List[Dict[str, Any]], text: str
) -> List[Dict[str, Any]]:
    """Promote ``Private Key`` entries from BEGIN-header-only to full PEM block.

    detect-secrets scans line-by-line, so for PEM keys it records only the
    ``-----BEGIN ... PRIVATE KEY-----`` armor header as the secret value.
    Downstream ``str.replace`` call sites would only strike that single line
    and leave the base64 body + END footer in the forwarded payload.  This
    helper finds enclosing PEM blocks in *text* and rewrites the secret
    value to the full block so the existing replace logic redacts the whole
    key.

    Because detect-secrets de-duplicates by secret value, multiple PEM
    blocks that share the same BEGIN header (e.g. two ``-----BEGIN PRIVATE
    KEY-----`` blocks in the same message) collapse to a single entry in
    *detected_secrets*.  In addition, ``_find_pem_blocks`` is intentionally
    broader than detect-secrets's built-in matcher and may catch blocks
    (e.g. ``BEGIN OPENSSH PRIVATE KEY``) that detect-secrets never flagged.
    To make sure every block is redacted, after expansion we add a
    synthetic ``Private Key`` entry for every PEM block in *text* that did
    not get claimed by an existing detected entry.

    Each PEM block is claimed at most once.  All other secret types pass
    through unchanged.
    """
    pem_blocks = _find_pem_blocks(text)
    claimed: set = set()

    expanded: List[Dict[str, Any]] = []
    for secret in detected_secrets:
        if secret.get("type") == "Private Key":
            header = secret.get("value")
            if header:
                for idx, block in enumerate(pem_blocks):
                    if idx in claimed:
                        continue
                    if header in block:
                        secret = {**secret, "value": block}
                        claimed.add(idx)
                        break
        expanded.append(secret)

    # Backfill every unclaimed PEM block that ``_find_pem_blocks`` returned.
    # detect-secrets de-duplicates entries by secret value, so multiple PEM
    # blocks that share the same BEGIN header collapse to one detected
    # record.  ``_find_pem_blocks`` is also intentionally broader than
    # detect-secrets's built-in matcher (it accepts headers like
    # ``BEGIN OPENSSH PRIVATE KEY`` and truncated blocks with no END
    # footer), so a block in the message may have no matching detected
    # entry at all.  Either way, the right behavior for ``hide_secrets``
    # is to redact the full block, so we synthesize a ``Private Key``
    # entry for every leftover.
    for idx, block in enumerate(pem_blocks):
        if idx in claimed:
            continue
        expanded.append({"type": "Private Key", "value": block})
        claimed.add(idx)

    return expanded


_custom_plugins_path = "file://" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "secrets_plugins"
)
_default_detect_secrets_config = {
    "plugins_used": [
        {"name": "SoftlayerDetector"},
        {"name": "StripeDetector"},
        {"name": "NpmDetector"},
        {"name": "IbmCosHmacDetector"},
        {"name": "DiscordBotTokenDetector"},
        {"name": "BasicAuthDetector"},
        {"name": "AzureStorageKeyDetector"},
        {"name": "ArtifactoryDetector"},
        {"name": "AWSKeyDetector"},
        {"name": "CloudantDetector"},
        {"name": "IbmCloudIamDetector"},
        {"name": "JwtTokenDetector"},
        {"name": "MailchimpDetector"},
        {"name": "SquareOAuthDetector"},
        {"name": "PrivateKeyDetector"},
        {"name": "TwilioKeyDetector"},
        {
            "name": "AdafruitKeyDetector",
            "path": _custom_plugins_path + "/adafruit.py",
        },
        {
            "name": "AdobeSecretDetector",
            "path": _custom_plugins_path + "/adobe.py",
        },
        {
            "name": "AgeSecretKeyDetector",
            "path": _custom_plugins_path + "/age_secret_key.py",
        },
        {
            "name": "AirtableApiKeyDetector",
            "path": _custom_plugins_path + "/airtable_api_key.py",
        },
        {
            "name": "AlgoliaApiKeyDetector",
            "path": _custom_plugins_path + "/algolia_api_key.py",
        },
        {
            "name": "AlibabaSecretDetector",
            "path": _custom_plugins_path + "/alibaba.py",
        },
        {
            "name": "AsanaSecretDetector",
            "path": _custom_plugins_path + "/asana.py",
        },
        {
            "name": "AtlassianApiTokenDetector",
            "path": _custom_plugins_path + "/atlassian_api_token.py",
        },
        {
            "name": "AuthressAccessKeyDetector",
            "path": _custom_plugins_path + "/authress_access_key.py",
        },
        {
            "name": "BittrexDetector",
            "path": _custom_plugins_path + "/beamer_api_token.py",
        },
        {
            "name": "BitbucketDetector",
            "path": _custom_plugins_path + "/bitbucket.py",
        },
        {
            "name": "BeamerApiTokenDetector",
            "path": _custom_plugins_path + "/bittrex.py",
        },
        {
            "name": "ClojarsApiTokenDetector",
            "path": _custom_plugins_path + "/clojars_api_token.py",
        },
        {
            "name": "CodecovAccessTokenDetector",
            "path": _custom_plugins_path + "/codecov_access_token.py",
        },
        {
            "name": "CoinbaseAccessTokenDetector",
            "path": _custom_plugins_path + "/coinbase_access_token.py",
        },
        {
            "name": "ConfluentDetector",
            "path": _custom_plugins_path + "/confluent.py",
        },
        {
            "name": "ContentfulApiTokenDetector",
            "path": _custom_plugins_path + "/contentful_api_token.py",
        },
        {
            "name": "DatabricksApiTokenDetector",
            "path": _custom_plugins_path + "/databricks_api_token.py",
        },
        {
            "name": "DatadogAccessTokenDetector",
            "path": _custom_plugins_path + "/datadog_access_token.py",
        },
        {
            "name": "DefinedNetworkingApiTokenDetector",
            "path": _custom_plugins_path + "/defined_networking_api_token.py",
        },
        {
            "name": "DigitaloceanDetector",
            "path": _custom_plugins_path + "/digitalocean.py",
        },
        {
            "name": "DopplerApiTokenDetector",
            "path": _custom_plugins_path + "/doppler_api_token.py",
        },
        {
            "name": "DroneciAccessTokenDetector",
            "path": _custom_plugins_path + "/droneci_access_token.py",
        },
        {
            "name": "DuffelApiTokenDetector",
            "path": _custom_plugins_path + "/duffel_api_token.py",
        },
        {
            "name": "DynatraceApiTokenDetector",
            "path": _custom_plugins_path + "/dynatrace_api_token.py",
        },
        {
            "name": "DiscordDetector",
            "path": _custom_plugins_path + "/discord.py",
        },
        {
            "name": "DropboxDetector",
            "path": _custom_plugins_path + "/dropbox.py",
        },
        {
            "name": "EasyPostDetector",
            "path": _custom_plugins_path + "/easypost.py",
        },
        {
            "name": "EtsyAccessTokenDetector",
            "path": _custom_plugins_path + "/etsy_access_token.py",
        },
        {
            "name": "FacebookAccessTokenDetector",
            "path": _custom_plugins_path + "/facebook_access_token.py",
        },
        {
            "name": "FastlyApiKeyDetector",
            "path": _custom_plugins_path + "/fastly_api_token.py",
        },
        {
            "name": "FinicityDetector",
            "path": _custom_plugins_path + "/finicity.py",
        },
        {
            "name": "FinnhubAccessTokenDetector",
            "path": _custom_plugins_path + "/finnhub_access_token.py",
        },
        {
            "name": "FlickrAccessTokenDetector",
            "path": _custom_plugins_path + "/flickr_access_token.py",
        },
        {
            "name": "FlutterwaveDetector",
            "path": _custom_plugins_path + "/flutterwave.py",
        },
        {
            "name": "FrameIoApiTokenDetector",
            "path": _custom_plugins_path + "/frameio_api_token.py",
        },
        {
            "name": "FreshbooksAccessTokenDetector",
            "path": _custom_plugins_path + "/freshbooks_access_token.py",
        },
        {
            "name": "GCPApiKeyDetector",
            "path": _custom_plugins_path + "/gcp_api_key.py",
        },
        {
            "name": "GitHubTokenCustomDetector",
            "path": _custom_plugins_path + "/github_token.py",
        },
        {
            "name": "GitLabDetector",
            "path": _custom_plugins_path + "/gitlab.py",
        },
        {
            "name": "GitterAccessTokenDetector",
            "path": _custom_plugins_path + "/gitter_access_token.py",
        },
        {
            "name": "GoCardlessApiTokenDetector",
            "path": _custom_plugins_path + "/gocardless_api_token.py",
        },
        {
            "name": "GrafanaDetector",
            "path": _custom_plugins_path + "/grafana.py",
        },
        {
            "name": "HashiCorpTFApiTokenDetector",
            "path": _custom_plugins_path + "/hashicorp_tf_api_token.py",
        },
        {
            "name": "HerokuApiKeyDetector",
            "path": _custom_plugins_path + "/heroku_api_key.py",
        },
        {
            "name": "HubSpotApiTokenDetector",
            "path": _custom_plugins_path + "/hubspot_api_key.py",
        },
        {
            "name": "HuggingFaceDetector",
            "path": _custom_plugins_path + "/huggingface.py",
        },
        {
            "name": "IntercomApiTokenDetector",
            "path": _custom_plugins_path + "/intercom_api_key.py",
        },
        {
            "name": "JFrogDetector",
            "path": _custom_plugins_path + "/jfrog.py",
        },
        {
            "name": "JWTBase64Detector",
            "path": _custom_plugins_path + "/jwt.py",
        },
        {
            "name": "KrakenAccessTokenDetector",
            "path": _custom_plugins_path + "/kraken_access_token.py",
        },
        {
            "name": "KucoinDetector",
            "path": _custom_plugins_path + "/kucoin.py",
        },
        {
            "name": "LaunchdarklyAccessTokenDetector",
            "path": _custom_plugins_path + "/launchdarkly_access_token.py",
        },
        {
            "name": "LinearDetector",
            "path": _custom_plugins_path + "/linear.py",
        },
        {
            "name": "LinkedInDetector",
            "path": _custom_plugins_path + "/linkedin.py",
        },
        {
            "name": "LobDetector",
            "path": _custom_plugins_path + "/lob.py",
        },
        {
            "name": "MailgunDetector",
            "path": _custom_plugins_path + "/mailgun.py",
        },
        {
            "name": "MapBoxApiTokenDetector",
            "path": _custom_plugins_path + "/mapbox_api_token.py",
        },
        {
            "name": "MattermostAccessTokenDetector",
            "path": _custom_plugins_path + "/mattermost_access_token.py",
        },
        {
            "name": "MessageBirdDetector",
            "path": _custom_plugins_path + "/messagebird.py",
        },
        {
            "name": "MicrosoftTeamsWebhookDetector",
            "path": _custom_plugins_path + "/microsoft_teams_webhook.py",
        },
        {
            "name": "NetlifyAccessTokenDetector",
            "path": _custom_plugins_path + "/netlify_access_token.py",
        },
        {
            "name": "NewRelicDetector",
            "path": _custom_plugins_path + "/new_relic.py",
        },
        {
            "name": "NYTimesAccessTokenDetector",
            "path": _custom_plugins_path + "/nytimes_access_token.py",
        },
        {
            "name": "OktaAccessTokenDetector",
            "path": _custom_plugins_path + "/okta_access_token.py",
        },
        {
            "name": "OpenAIApiKeyDetector",
            "path": _custom_plugins_path + "/openai_api_key.py",
        },
        {
            "name": "PlanetScaleDetector",
            "path": _custom_plugins_path + "/planetscale.py",
        },
        {
            "name": "PostmanApiTokenDetector",
            "path": _custom_plugins_path + "/postman_api_token.py",
        },
        {
            "name": "PrefectApiTokenDetector",
            "path": _custom_plugins_path + "/prefect_api_token.py",
        },
        {
            "name": "PulumiApiTokenDetector",
            "path": _custom_plugins_path + "/pulumi_api_token.py",
        },
        {
            "name": "PyPiUploadTokenDetector",
            "path": _custom_plugins_path + "/pypi_upload_token.py",
        },
        {
            "name": "RapidApiAccessTokenDetector",
            "path": _custom_plugins_path + "/rapidapi_access_token.py",
        },
        {
            "name": "ReadmeApiTokenDetector",
            "path": _custom_plugins_path + "/readme_api_token.py",
        },
        {
            "name": "RubygemsApiTokenDetector",
            "path": _custom_plugins_path + "/rubygems_api_token.py",
        },
        {
            "name": "ScalingoApiTokenDetector",
            "path": _custom_plugins_path + "/scalingo_api_token.py",
        },
        {
            "name": "SendbirdDetector",
            "path": _custom_plugins_path + "/sendbird.py",
        },
        {
            "name": "SendGridApiTokenDetector",
            "path": _custom_plugins_path + "/sendgrid_api_token.py",
        },
        {
            "name": "SendinBlueApiTokenDetector",
            "path": _custom_plugins_path + "/sendinblue_api_token.py",
        },
        {
            "name": "SentryAccessTokenDetector",
            "path": _custom_plugins_path + "/sentry_access_token.py",
        },
        {
            "name": "ShippoApiTokenDetector",
            "path": _custom_plugins_path + "/shippo_api_token.py",
        },
        {
            "name": "ShopifyDetector",
            "path": _custom_plugins_path + "/shopify.py",
        },
        {
            "name": "SlackDetector",
            "path": _custom_plugins_path + "/slack.py",
        },
        {
            "name": "SnykApiTokenDetector",
            "path": _custom_plugins_path + "/snyk_api_token.py",
        },
        {
            "name": "SquarespaceAccessTokenDetector",
            "path": _custom_plugins_path + "/squarespace_access_token.py",
        },
        {
            "name": "SumoLogicDetector",
            "path": _custom_plugins_path + "/sumologic.py",
        },
        {
            "name": "TelegramBotApiTokenDetector",
            "path": _custom_plugins_path + "/telegram_bot_api_token.py",
        },
        {
            "name": "TravisCiAccessTokenDetector",
            "path": _custom_plugins_path + "/travisci_access_token.py",
        },
        {
            "name": "TwitchApiTokenDetector",
            "path": _custom_plugins_path + "/twitch_api_token.py",
        },
        {
            "name": "TwitterDetector",
            "path": _custom_plugins_path + "/twitter.py",
        },
        {
            "name": "TypeformApiTokenDetector",
            "path": _custom_plugins_path + "/typeform_api_token.py",
        },
        {
            "name": "VaultDetector",
            "path": _custom_plugins_path + "/vault.py",
        },
        {
            "name": "YandexDetector",
            "path": _custom_plugins_path + "/yandex.py",
        },
        {
            "name": "ZendeskSecretKeyDetector",
            "path": _custom_plugins_path + "/zendesk_secret_key.py",
        },
        {"name": "Base64HighEntropyString", "limit": 3.0},
        {"name": "HexHighEntropyString", "limit": 3.0},
    ]
}


class _ENTERPRISE_SecretDetection(CustomGuardrail):
    def __init__(self, detect_secrets_config: Optional[dict] = None, **kwargs):
        self.user_defined_detect_secrets_config = detect_secrets_config
        super().__init__(**kwargs)

    def scan_message_for_secrets(self, message_content: str):
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import transient_settings

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(message_content.encode("utf-8"))
        temp_file.close()

        secrets = SecretsCollection()

        detect_secrets_config = (
            self.user_defined_detect_secrets_config or _default_detect_secrets_config
        )
        with transient_settings(detect_secrets_config):
            secrets.scan_file(temp_file.name)

        os.remove(temp_file.name)

        detected_secrets = []
        for file in secrets.files:
            for found_secret in secrets[file]:
                if found_secret.secret_value is None:
                    continue
                detected_secrets.append(
                    {"type": found_secret.type, "value": found_secret.secret_value}
                )

        # detect-secrets is line-based: for PEM private keys it records only
        # the BEGIN armor header as the secret value.  Expand each ``Private
        # Key`` entry to the full PEM block so that downstream ``str.replace``
        # call sites redact the entire key (body + END footer), not just the
        # first line.
        detected_secrets = _expand_private_key_values(
            detected_secrets, message_content
        )

        return detected_secrets

    async def should_run_check(self, user_api_key_dict: UserAPIKeyAuth) -> bool:
        if user_api_key_dict.permissions is not None:
            if GUARDRAIL_NAME in user_api_key_dict.permissions:
                if user_api_key_dict.permissions[GUARDRAIL_NAME] is False:
                    return False

        return True

    #### CALL HOOKS - proxy only ####
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,  # "completion", "embeddings", "image_generation", "moderation"
    ):
        if await self.should_run_check(user_api_key_dict) is False:
            return

        # Covers multimodal list content + Responses-API input.
        def _redact_message_text(text: str) -> str:
            detected_secrets = self.scan_message_for_secrets(text)
            for secret in detected_secrets:
                text = text.replace(secret["value"], "[REDACTED]")
            if detected_secrets:
                secret_types = [secret["type"] for secret in detected_secrets]
                verbose_proxy_logger.warning(
                    f"Detected and redacted secrets in message: {secret_types}"
                )
            return text

        walk_user_text(data, _redact_message_text)

        if "prompt" in data:
            if isinstance(data["prompt"], str):
                detected_secrets = self.scan_message_for_secrets(data["prompt"])
                for secret in detected_secrets:
                    data["prompt"] = data["prompt"].replace(
                        secret["value"], "[REDACTED]"
                    )
                if len(detected_secrets) > 0:
                    secret_types = [secret["type"] for secret in detected_secrets]
                    verbose_proxy_logger.warning(
                        f"Detected and redacted secrets in prompt: {secret_types}"
                    )
            elif isinstance(data["prompt"], list):
                # Index back into the list — assigning to ``item`` would only
                # rebind the loop variable and leave ``data["prompt"]``
                # carrying the unredacted secret.
                for idx, item in enumerate(data["prompt"]):
                    if isinstance(item, str):
                        detected_secrets = self.scan_message_for_secrets(item)
                        for secret in detected_secrets:
                            item = item.replace(secret["value"], "[REDACTED]")
                        data["prompt"][idx] = item
                        if len(detected_secrets) > 0:
                            secret_types = [
                                secret["type"] for secret in detected_secrets
                            ]
                            verbose_proxy_logger.warning(
                                f"Detected and redacted secrets in prompt: {secret_types}"
                            )

        # ``data["input"]`` (Responses API and embeddings/moderation) is
        # already covered by ``walk_user_text`` above.
        return
