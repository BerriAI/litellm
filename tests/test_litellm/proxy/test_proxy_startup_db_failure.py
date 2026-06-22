import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

@pytest.mark.asyncio
async def test_proxy_startup_db_failure_reproduction(monkeypatch):
    """
    Test for litellm DB connection failure during startup.
    When Litellm starts up and the database is unstable/down, 
    the query for general_settings in initialize_scheduled_background_jobs 
    throws an exception, causing store_model_in_db to remain False.
    This starts a background retry task. Upon DB recovery, the retry task 
    registers the add_deployment and get_credentials jobs.
    
    The test asserts that both 'add_deployment_job' and 'get_credentials_job' 
    are registered after the background retry task runs and succeeds.
    """
    # Delete environment overrides so we rely on DB check
    monkeypatch.delenv("STORE_MODEL_IN_DB", raising=False)

    from litellm.proxy.proxy_server import ProxyStartupEvent
    from litellm.proxy.utils import ProxyLogging

    # 1. Mock DB connection to throw an exception on initial call, but succeed on retry
    mock_prisma_client = MagicMock()
    mock_db_record = MagicMock()
    mock_db_record.param_value = {"store_model_in_db": True}

    # Side effect: first call (startup check) fails; second call (retry loop) succeeds
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(
        side_effect=[
            Exception("Database Connection Failed / Connection Timed Out"),
            mock_db_record
        ]
    )

    # 2. Mock proxy logging and proxy config
    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.slack_alerting_instance = MagicMock()
    
    mock_proxy_config = AsyncMock()

    # 3. Patch AsyncIOScheduler to spy/mock scheduler calls
    with patch("litellm.proxy.proxy_server.AsyncIOScheduler") as mock_scheduler_class:
        mock_scheduler_instance = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler_instance

        # Patch proxy_config and initialize store_model_in_db to False (default YAML config behavior)
        with (
            patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", False),
            patch("litellm.proxy.proxy_server.get_secret_bool", return_value=False),
        ):
            # 4. Mock asyncio.sleep to fast-forward the 5 seconds sleep in retry loop
            original_sleep = asyncio.sleep
            async def mock_sleep(seconds, *args, **kwargs):
                if seconds == 5:
                    await original_sleep(0.01)
                else:
                    await original_sleep(seconds)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                # 5. Invoke initialize_scheduled_background_jobs
                await ProxyStartupEvent.initialize_scheduled_background_jobs(
                    general_settings={"disable_spend_logs": True},
                    prisma_client=mock_prisma_client,
                    proxy_budget_rescheduler_min_time=1,
                    proxy_budget_rescheduler_max_time=2,
                    proxy_batch_write_at=5,
                    proxy_logging_obj=mock_proxy_logging,
                )

                # 6. Yield execution to allow the background retry task to run and complete
                await asyncio.sleep(0.1)

            # 7. Assert that the add_deployment and get_credentials background jobs ARE registered.
            add_deployment_job_registered = False
            get_credentials_job_registered = False

            for call_args in mock_scheduler_instance.add_job.call_args_list:
                job_id = call_args.kwargs.get("id")
                if job_id == "add_deployment_job":
                    add_deployment_job_registered = True
                elif job_id == "get_credentials_job":
                    get_credentials_job_registered = True

            assert add_deployment_job_registered, "add_deployment_job was not registered after database recovery."
            assert get_credentials_job_registered, "get_credentials_job was not registered after database recovery."
