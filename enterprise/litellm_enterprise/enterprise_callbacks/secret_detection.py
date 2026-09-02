# +-------------------------------------------------------------+
#
#           Use SecretDetection /moderations for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import functools
import tempfile
from contextvars import ContextVar
from typing import TYPE_CHECKING, ClassVar, Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import walk_user_text
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "hide_secrets"

GUARDRAIL_PROVIDER = "hide-secrets"

# Per-invocation tally of redacted secrets by detect-secrets plugin type; None
# means the guardrail did not run, so _process_response records nothing.
_masked_entity_count: ContextVar[Optional[dict]] = ContextVar(
    "hide_secrets_masked_entity_count", default=None
)

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
    # Keeps proxied traffic on async_pre_call_hook (the unified apply_guardrail
    # path skips should_run_check and never sees data["prompt"]).
    use_native_lifecycle_hooks: ClassVar[bool] = True

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

        return detected_secrets

    def redact_text(self, text: str, source: str = "message") -> str:
        """Replace every detected secret in ``text`` with ``[REDACTED]`` and
        tally the detected types into the per-invocation masked-entity count."""
        detected_secrets = self.scan_message_for_secrets(text)
        if not detected_secrets:
            return text
        counts = _masked_entity_count.get()
        if counts is not None:
            for secret in detected_secrets:
                counts[secret["type"]] = counts.get(secret["type"], 0) + 1
        secret_types = [secret["type"] for secret in detected_secrets]
        verbose_proxy_logger.warning(
            f"Detected and redacted secrets in {source}: {secret_types}"
        )
        return functools.reduce(
            lambda redacted, secret: redacted.replace(secret["value"], "[REDACTED]"),
            detected_secrets,
            text,
        )

    async def should_run_check(self, user_api_key_dict: UserAPIKeyAuth) -> bool:
        if user_api_key_dict.permissions is not None:
            if GUARDRAIL_NAME in user_api_key_dict.permissions:
                if user_api_key_dict.permissions[GUARDRAIL_NAME] is False:
                    return False

        return True

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Unified-interface entrypoint, used by /guardrails/apply_guardrail
        (the UI test playground). Proxied traffic keeps using
        ``async_pre_call_hook``, see ``use_native_lifecycle_hooks``."""
        texts = inputs.get("texts")
        if not texts:
            return inputs
        _masked_entity_count.set({})
        return {**inputs, "texts": [self.redact_text(text) for text in texts]}

    #### CALL HOOKS - proxy only ####
    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,  # "completion", "embeddings", "image_generation", "moderation"
    ):
        _masked_entity_count.set(None)
        if await self.should_run_check(user_api_key_dict) is False:
            return

        _masked_entity_count.set({})

        # Covers multimodal list content + Responses-API input.
        walk_user_text(data, self.redact_text)

        if "prompt" in data:
            if isinstance(data["prompt"], str):
                data["prompt"] = self.redact_text(data["prompt"], source="prompt")
            elif isinstance(data["prompt"], list):
                for idx, item in enumerate(data["prompt"]):
                    if isinstance(item, str):
                        data["prompt"][idx] = self.redact_text(item, source="prompt")

        return

    def _process_response(
        self,
        response: Optional[dict],
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        event_type: Optional[GuardrailEventHooks] = None,
        original_inputs: Optional[dict] = None,
    ):
        """Record allow/mask plus the masked-entity tally for a completed run.

        Records nothing when the guardrail inspected nothing (opted-out key,
        empty inputs) or when the instance has no guardrail_name (legacy
        ``litellm_settings.callbacks`` deployments, which predate guardrail
        telemetry and stay without it).
        """
        counts = _masked_entity_count.get()
        _masked_entity_count.set(None)
        if counts is None or self.guardrail_name is None:
            return response
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response="mask" if counts else "allow",
            request_data=request_data,
            guardrail_status="success",
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            guardrail_provider=GUARDRAIL_PROVIDER,
            masked_entity_count=counts,
        )
        return response

    def _process_error(
        self,
        e: Exception,
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        event_type: Optional[GuardrailEventHooks] = None,
    ):
        """Label the failed run with this guardrail's provider so error rows
        group with the successful ones in the monitor. Nameless legacy
        instances record nothing, matching ``_process_response``."""
        _masked_entity_count.set(None)
        if self.guardrail_name is None:
            raise e
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=e,
            request_data=request_data,
            guardrail_status=(
                "guardrail_intervened"
                if self._is_guardrail_intervention(e)
                else "guardrail_failed_to_respond"
            ),
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            guardrail_provider=GUARDRAIL_PROVIDER,
        )
        raise e
