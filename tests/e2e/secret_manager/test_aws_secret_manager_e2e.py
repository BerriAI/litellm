import hashlib
from typing import Final

import pytest
from e2e_metadata import Domain, Subject, meta
from pydantic import TypeAdapter

from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.secret_managers.main import get_secret

pytestmark = pytest.mark.e2e

_AWS_FIXTURE_MASTER_KEY_SHA256: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
_STRING_MAP: Final = TypeAdapter(dict[str, str])


@meta(Subject(domain=Domain.DEPLOY_OPS))
def test_aws_secret_manager() -> None:
    AWSSecretsManagerV2.load_aws_secret_manager(use_aws_secret_manager=True)
    secret_value: Final = _STRING_MAP.validate_json(get_secret("litellm_master_key"))
    assert (
        hashlib.sha256(secret_value["litellm_master_key"].encode()).hexdigest()
        == _AWS_FIXTURE_MASTER_KEY_SHA256
    )
