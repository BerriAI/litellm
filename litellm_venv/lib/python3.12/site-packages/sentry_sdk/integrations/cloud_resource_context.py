import json
import urllib3

from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict


CONTEXT_TYPE = "cloud_resource"

AWS_METADATA_HOST = "169.254.169.254"
AWS_TOKEN_URL = "http://{}/latest/api/token".format(AWS_METADATA_HOST)
AWS_METADATA_URL = "http://{}/latest/dynamic/instance-identity/document".format(
    AWS_METADATA_HOST
)

GCP_METADATA_HOST = "metadata.google.internal"
GCP_METADATA_URL = "http://{}/computeMetadata/v1/?recursive=true".format(
    GCP_METADATA_HOST
)


class CLOUD_PROVIDER:  # noqa: N801
    """
    Name of the cloud provider.
    see https://opentelemetry.io/docs/reference/specification/resource/semantic_conventions/cloud/
    """

    ALIBABA = "alibaba_cloud"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    IBM = "ibm_cloud"
    TENCENT = "tencent_cloud"


class CLOUD_PLATFORM:  # noqa: N801
    """
    The cloud platform.
    see https://opentelemetry.io/docs/reference/specification/resource/semantic_conventions/cloud/
    """

    AWS_EC2 = "aws_ec2"
    GCP_COMPUTE_ENGINE = "gcp_compute_engine"


class CloudResourceContextIntegration(Integration):
    """
    Adds cloud resource context to the Senty scope
    """

    identifier = "cloudresourcecontext"

    cloud_provider = ""

    aws_token = ""
    http = urllib3.PoolManager()

    gcp_metadata = None

    def __init__(self, cloud_provider=""):
        # type: (str) -> None
        CloudResourceContextIntegration.cloud_provider = cloud_provider

    @classmethod
    def _is_aws(cls):
        # type: () -> bool
        try:
            r = cls.http.request(
                "PUT",
                AWS_TOKEN_URL,
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            )

            if r.status != 200:
                return False

            cls.aws_token = r.data.decode()
            return True

        except Exception:
            return False

    @classmethod
    def _get_aws_context(cls):
        # type: () -> Dict[str, str]
        ctx = {
            "cloud.provider": CLOUD_PROVIDER.AWS,
            "cloud.platform": CLOUD_PLATFORM.AWS_EC2,
        }

        try:
            r = cls.http.request(
                "GET",
                AWS_METADATA_URL,
                headers={"X-aws-ec2-metadata-token": cls.aws_token},
            )

            if r.status != 200:
                return ctx

            data = json.loads(r.data.decode("utf-8"))

            try:
                ctx["cloud.account.id"] = data["accountId"]
            except Exception:
                pass

            try:
                ctx["cloud.availability_zone"] = data["availabilityZone"]
            except Exception:
                pass

            try:
                ctx["cloud.region"] = data["region"]
            except Exception:
                pass

            try:
                ctx["host.id"] = data["instanceId"]
            except Exception:
                pass

            try:
                ctx["host.type"] = data["instanceType"]
            except Exception:
                pass

        except Exception:
            pass

        return ctx

    @classmethod
    def _is_gcp(cls):
        # type: () -> bool
        try:
            r = cls.http.request(
                "GET",
                GCP_METADATA_URL,
                headers={"Metadata-Flavor": "Google"},
            )

            if r.status != 200:
                return False

            cls.gcp_metadata = json.loads(r.data.decode("utf-8"))
            return True

        except Exception:
            return False

    @classmethod
    def _get_gcp_context(cls):
        # type: () -> Dict[str, str]
        ctx = {
            "cloud.provider": CLOUD_PROVIDER.GCP,
            "cloud.platform": CLOUD_PLATFORM.GCP_COMPUTE_ENGINE,
        }

        try:
            if cls.gcp_metadata is None:
                r = cls.http.request(
                    "GET",
                    GCP_METADATA_URL,
                    headers={"Metadata-Flavor": "Google"},
                )

                if r.status != 200:
                    return ctx

                cls.gcp_metadata = json.loads(r.data.decode("utf-8"))

            try:
                ctx["cloud.account.id"] = cls.gcp_metadata["project"]["projectId"]
            except Exception:
                pass

            try:
                ctx["cloud.availability_zone"] = cls.gcp_metadata["instance"][
                    "zone"
                ].split("/")[-1]
            except Exception:
                pass

            try:
                # only populated in google cloud run
                ctx["cloud.region"] = cls.gcp_metadata["instance"]["region"].split("/")[
                    -1
                ]
            except Exception:
                pass

            try:
                ctx["host.id"] = cls.gcp_metadata["instance"]["id"]
            except Exception:
                pass

        except Exception:
            pass

        return ctx

    @classmethod
    def _get_cloud_provider(cls):
        # type: () -> str
        if cls._is_aws():
            return CLOUD_PROVIDER.AWS

        if cls._is_gcp():
            return CLOUD_PROVIDER.GCP

        return ""

    @classmethod
    def _get_cloud_resource_context(cls):
        # type: () -> Dict[str, str]
        cloud_provider = (
            cls.cloud_provider
            if cls.cloud_provider != ""
            else CloudResourceContextIntegration._get_cloud_provider()
        )
        if cloud_provider in context_getters.keys():
            return context_getters[cloud_provider]()

        return {}

    @staticmethod
    def setup_once():
        # type: () -> None
        cloud_provider = CloudResourceContextIntegration.cloud_provider
        unsupported_cloud_provider = (
            cloud_provider != "" and cloud_provider not in context_getters.keys()
        )

        if unsupported_cloud_provider:
            logger.warning(
                "Invalid value for cloud_provider: %s (must be in %s). Falling back to autodetection...",
                CloudResourceContextIntegration.cloud_provider,
                list(context_getters.keys()),
            )

        context = CloudResourceContextIntegration._get_cloud_resource_context()
        if context != {}:
            set_context(CONTEXT_TYPE, context)


# Map with the currently supported cloud providers
# mapping to functions extracting the context
context_getters = {
    CLOUD_PROVIDER.AWS: CloudResourceContextIntegration._get_aws_context,
    CLOUD_PROVIDER.GCP: CloudResourceContextIntegration._get_gcp_context,
}
