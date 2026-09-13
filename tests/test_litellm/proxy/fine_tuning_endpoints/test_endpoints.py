"""
require_managed_files enforcement for litellm/proxy/fine_tuning_endpoints/endpoints.py

Ownership rows only exist for LiteLLM managed ids. A raw provider id sent to these
routes is forwarded to the provider under the shared proxy credentials with no tenant
check, so any caller who learns another tenant's file id can train on it, and any
caller who learns another tenant's job id can read or cancel it.

Each test asserts BOTH that the request is rejected AND that every downstream provider
seam stayed untouched, so a guard that raises after the provider call would still fail.
"""

import base64
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from fastapi import Response

import litellm
import litellm.proxy.fine_tuning_endpoints.endpoints as endpoints
import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.llms.openai import LiteLLMFineTuningJobCreate
from litellm.types.utils import LiteLLMFineTuningJob, SpecialEnums

RAW_FILE_ID = "file-victim-abc123"
RAW_JOB_ID = "ftjob-victim-abc123"


def _unified_file_id() -> str:
    unified = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json", "victim-unified-id", "gpt-4o-mini", RAW_FILE_ID, "gpt-4o-mini-id"
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


def _unified_job_id() -> str:
    unified = SpecialEnums.LITELLM_MANAGED_GENERIC_RESPONSE_COMPLETE_STR.value.format("gpt-4o-mini-id", RAW_JOB_ID)
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


def _job() -> LiteLLMFineTuningJob:
    job = LiteLLMFineTuningJob(
        id=RAW_JOB_ID,
        created_at=1234567890,
        fine_tuned_model=None,
        finished_at=None,
        hyperparameters={"n_epochs": 1},
        model="gpt-4o-mini",
        object="fine_tuning.job",
        organization_id="org-test",
        result_files=[],
        seed=0,
        status="running",
        trained_tokens=None,
        training_file=RAW_FILE_ID,
        validation_file=None,
    )
    job._hidden_params = {}
    return job


class FakeRequest:
    def __init__(self):
        self.headers = {}
        self.query_params = {}

    async def json(self):
        return {}


@dataclass(frozen=True)
class ManagedResourceAccessCheckerStub:
    file_access: bool = True
    object_access: bool = True

    async def can_user_call_unified_file_id(
        self,
        unified_file_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        return self.file_access

    async def can_user_call_unified_object_id(
        self,
        unified_object_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        return self.object_access


class Seams:
    def __init__(self, router: MagicMock, litellm_calls: dict[str, AsyncMock], logging: MagicMock):
        self.router = router
        self.litellm_calls = litellm_calls
        self.logging = logging

    def assert_no_provider_call(self) -> None:
        for name, mock in self.litellm_calls.items():
            assert mock.call_count == 0, f"litellm.{name} was called"
        for name in ("acreate_fine_tuning_job", "aretrieve_fine_tuning_job", "acancel_fine_tuning_job"):
            assert getattr(self.router, name).call_count == 0, f"router.{name} was called"


@pytest.fixture
def seams():
    logging = MagicMock(spec=ProxyLogging)
    logging.post_call_success_hook = AsyncMock(side_effect=lambda **kw: kw["response"])
    logging.post_call_failure_hook = AsyncMock()
    logging.update_request_status = AsyncMock()
    logging.get_proxy_hook = MagicMock(return_value=None)

    router = MagicMock(spec=Router)
    router.acreate_fine_tuning_job = AsyncMock(return_value=_job())
    router.aretrieve_fine_tuning_job = AsyncMock(return_value=_job())
    router.acancel_fine_tuning_job = AsyncMock(return_value=_job())

    litellm_calls = {
        name: AsyncMock(return_value=_job())
        for name in ("acreate_fine_tuning_job", "aretrieve_fine_tuning_job", "acancel_fine_tuning_job")
    }

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "common_processing_pre_call_logic",
                AsyncMock(side_effect=lambda self=None, **kw: (self.data if self else {}, MagicMock())),
            )
        )
        stack.enter_context(patch.object(ProxyBaseLLMRequestProcessing, "get_custom_headers", MagicMock(return_value={})))
        for name, mock in litellm_calls.items():
            stack.enter_context(patch.object(litellm, name, mock))
        stack.enter_context(patch.object(proxy_server, "llm_router", router))
        stack.enter_context(patch.object(proxy_server, "proxy_logging_obj", logging))
        stack.enter_context(patch.object(proxy_server, "premium_user", True))
        stack.enter_context(patch.object(proxy_server, "general_settings", {}))
        stack.enter_context(patch.object(proxy_server, "proxy_config", MagicMock()))
        stack.enter_context(patch.object(proxy_server, "version", "test-version"))
        stack.enter_context(patch.object(endpoints, "fine_tuning_config", [{"custom_llm_provider": "openai"}]))
        yield Seams(router=router, litellm_calls=litellm_calls, logging=logging)


async def _create(training_file: str, validation_file: str | None = None):
    return await endpoints.create_fine_tuning_job(
        request=FakeRequest(),
        fastapi_response=Response(),
        fine_tuning_request=LiteLLMFineTuningJobCreate(
            model="gpt-4o-mini",
            training_file=training_file,
            validation_file=validation_file,
            custom_llm_provider="openai",
        ),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )


async def _retrieve(job_id: str):
    return await endpoints.retrieve_fine_tuning_job(
        request=FakeRequest(),
        fastapi_response=Response(),
        fine_tuning_job_id=job_id,
        custom_llm_provider="openai",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )


async def _cancel(job_id: str):
    return await endpoints.cancel_fine_tuning_job(
        request=FakeRequest(),
        fastapi_response=Response(),
        fine_tuning_job_id=job_id,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )


@pytest.mark.asyncio
async def test_create__raw_training_file_rejected_when_managed_files_required(seams):
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(ProxyException) as exc:
            await _create(RAW_FILE_ID)

    assert exc.value.code == "400"
    seams.assert_no_provider_call()


@pytest.mark.asyncio
async def test_create__raw_validation_file_rejected_when_managed_files_required(seams):
    """The validation file is uploaded and readable exactly like the training file,
    so a managed training_file must not smuggle a raw validation_file past the guard."""
    seams.logging.get_proxy_hook.return_value = ManagedResourceAccessCheckerStub()

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(ProxyException) as exc:
            await _create(_unified_file_id(), validation_file=RAW_FILE_ID)

    assert exc.value.code == "400"
    seams.assert_no_provider_call()


@pytest.mark.asyncio
async def test_create__unified_training_file_allowed_when_managed_files_required(seams):
    seams.logging.get_proxy_hook.return_value = ManagedResourceAccessCheckerStub()

    with patch.object(litellm, "require_managed_files", True):
        await _create(_unified_file_id())

    assert seams.router.acreate_fine_tuning_job.call_count == 1


@pytest.mark.asyncio
async def test_create__other_teams_unified_training_file_rejected(seams):
    seams.logging.get_proxy_hook.return_value = ManagedResourceAccessCheckerStub(file_access=False)

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(ProxyException) as exc:
            await _create(_unified_file_id())

    assert exc.value.code == "403"
    seams.assert_no_provider_call()


@pytest.mark.asyncio
async def test_create__raw_training_file_allowed_when_managed_files_not_required(seams):
    with patch.object(litellm, "require_managed_files", False):
        await _create(RAW_FILE_ID)

    assert seams.litellm_calls["acreate_fine_tuning_job"].call_count == 1


@pytest.mark.asyncio
async def test_retrieve__raw_job_id_rejected_when_managed_files_required(seams):
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(ProxyException) as exc:
            await _retrieve(RAW_JOB_ID)

    assert exc.value.code == "400"
    seams.assert_no_provider_call()


@pytest.mark.asyncio
async def test_retrieve__unified_job_id_allowed_when_managed_files_required(seams):
    seams.logging.get_proxy_hook.return_value = ManagedResourceAccessCheckerStub()

    with patch.object(litellm, "require_managed_files", True):
        await _retrieve(_unified_job_id())

    assert seams.router.aretrieve_fine_tuning_job.call_count == 1


@pytest.mark.asyncio
async def test_cancel__raw_job_id_rejected_when_managed_files_required(seams):
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(ProxyException) as exc:
            await _cancel(RAW_JOB_ID)

    assert exc.value.code == "400"
    seams.assert_no_provider_call()


@pytest.mark.asyncio
async def test_cancel__unified_job_id_allowed_when_managed_files_required(seams):
    seams.logging.get_proxy_hook.return_value = ManagedResourceAccessCheckerStub()

    with patch.object(litellm, "require_managed_files", True):
        await _cancel(_unified_job_id())

    assert seams.router.acancel_fine_tuning_job.call_count == 1
