import asyncio
import logging
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from litellm.proxy.common_utils.load_config_utils import (
    gcs_config_bucket,
    get_config_from_bucket,
    get_file_contents_from_s3,
    resolve_bucket_includes,
)
from tests._master_key import MASTER_KEY


class TestGetFileContentsFromS3:
    """Test suite for S3 config loading functionality."""

    @patch("boto3.client")
    @patch("litellm.main.bedrock_converse_chat_completion")
    def test_get_file_contents_from_s3_no_temp_file_creation(
        self,
        mock_bedrock: MagicMock,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_credentials: Final = MagicMock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_token"
        mock_bedrock.get_credentials.return_value = mock_credentials
        mock_s3_client: Final = mock_boto3_client.return_value
        expected_config: Final = {
            "model_list": [{"model_name": "assistant", "litellm_params": {"model": "test-model"}}],
            "extension": {"disabled": False, "zero": 0, "empty": None},
        }
        mock_s3_client.get_object.return_value = {"Body": BytesIO(yaml.safe_dump(expected_config).encode("utf-8"))}

        result: Final = get_file_contents_from_s3("test-bucket", "config.yaml")

        assert result == expected_config
        assert tuple(tmp_path.iterdir()) == ()
        mock_boto3_client.assert_called_once_with(
            "s3",
            aws_access_key_id=mock_credentials.access_key,
            aws_secret_access_key=mock_credentials.secret_key,
            aws_session_token=mock_credentials.token,
        )
        mock_s3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="config.yaml")


class TestBucketConfigIncludes:

    @staticmethod
    def _bucket(objects):
        async def fetch(object_key):
            return objects.get(object_key)

        return fetch

    @pytest.mark.asyncio
    async def test_include_resolves_against_the_config_objects_prefix(self):
        merged = await resolve_bucket_includes(
            config={"include": ["model_config.yaml"], "general_settings": {"master_key": MASTER_KEY}},
            object_key="configs/prod/config.yaml",
            fetch=self._bucket(
                {"configs/prod/model_config.yaml": {"model_list": [{"model_name": "gpt-4o-mini"}]}}
            ),
        )

        assert merged == {
            "general_settings": {"master_key": MASTER_KEY},
            "model_list": [{"model_name": "gpt-4o-mini"}],
        }

    @pytest.mark.asyncio
    async def test_include_with_a_leading_slash_reads_from_the_bucket_root(self):
        merged = await resolve_bucket_includes(
            config={"include": ["/shared/models.yaml"]},
            object_key="configs/prod/config.yaml",
            fetch=self._bucket({"shared/models.yaml": {"model_list": [{"model_name": "shared"}]}}),
        )

        assert merged == {"model_list": [{"model_name": "shared"}]}

    @pytest.mark.asyncio
    async def test_include_walks_out_of_the_prefix_with_dot_dot(self):
        merged = await resolve_bucket_includes(
            config={"include": ["../shared/models.yaml"]},
            object_key="configs/prod/config.yaml",
            fetch=self._bucket({"configs/shared/models.yaml": {"model_list": [{"model_name": "shared"}]}}),
        )

        assert merged == {"model_list": [{"model_name": "shared"}]}

    @pytest.mark.asyncio
    async def test_included_configs_may_declare_further_includes(self):
        merged = await resolve_bucket_includes(
            config={"include": ["models.yaml"]},
            object_key="configs/config.yaml",
            fetch=self._bucket(
                {
                    "configs/models.yaml": {
                        "include": ["extra/more_models.yaml"],
                        "model_list": [{"model_name": "first"}],
                    },
                    "configs/extra/more_models.yaml": {"model_list": [{"model_name": "second"}]},
                }
            ),
        )

        assert merged == {"model_list": [{"model_name": "first"}, {"model_name": "second"}]}

    @pytest.mark.asyncio
    async def test_a_nested_include_resolves_against_the_object_that_declares_it(self):
        merged = await resolve_bucket_includes(
            config={"include": ["shared/models.yaml"]},
            object_key="configs/config.yaml",
            fetch=self._bucket(
                {
                    "configs/shared/models.yaml": {
                        "include": ["more_models.yaml"],
                        "model_list": [{"model_name": "first"}],
                    },
                    "configs/shared/more_models.yaml": {"model_list": [{"model_name": "second"}]},
                    "configs/more_models.yaml": {"model_list": [{"model_name": "wrong-prefix"}]},
                }
            ),
        )

        assert merged == {"model_list": [{"model_name": "first"}, {"model_name": "second"}]}

    @pytest.mark.asyncio
    async def test_an_object_pulled_in_twice_is_merged_once(self):
        merged = await resolve_bucket_includes(
            config={"include": ["a.yaml", "b.yaml"]},
            object_key="configs/config.yaml",
            fetch=self._bucket(
                {
                    "configs/a.yaml": {"include": ["shared.yaml"]},
                    "configs/b.yaml": {"include": ["./shared.yaml"]},
                    "configs/shared.yaml": {"model_list": [{"model_name": "shared"}]},
                }
            ),
        )

        assert merged == {"model_list": [{"model_name": "shared"}]}

    @pytest.mark.asyncio
    async def test_a_cycle_between_included_objects_terminates(self):
        merged = await asyncio.wait_for(
            resolve_bucket_includes(
                config={"include": ["a.yaml"]},
                object_key="configs/config.yaml",
                fetch=self._bucket(
                    {
                        "configs/a.yaml": {"include": ["b.yaml"], "model_list": [{"model_name": "from-a"}]},
                        "configs/b.yaml": {"include": ["a.yaml"], "model_list": [{"model_name": "from-b"}]},
                    }
                ),
            ),
            timeout=10,
        )

        assert merged == {"model_list": [{"model_name": "from-a"}, {"model_name": "from-b"}]}

    @pytest.mark.asyncio
    async def test_list_values_are_extended_and_other_values_are_overridden(self):
        merged = await resolve_bucket_includes(
            config={
                "include": ["models.yaml"],
                "model_list": [{"model_name": "from-root"}],
                "litellm_settings": {"drop_params": True},
            },
            object_key="config.yaml",
            fetch=self._bucket(
                {
                    "models.yaml": {
                        "model_list": [{"model_name": "from-include"}],
                        "litellm_settings": {"num_retries": 3},
                    }
                }
            ),
        )

        assert merged == {
            "model_list": [{"model_name": "from-root"}, {"model_name": "from-include"}],
            "litellm_settings": {"num_retries": 3},
        }

    @pytest.mark.asyncio
    async def test_a_missing_included_object_fails_loudly_with_its_key(self):
        with pytest.raises(FileNotFoundError, match=re.escape("configs/prod/model_config.yaml")):
            await resolve_bucket_includes(
                config={"include": ["model_config.yaml"]},
                object_key="configs/prod/config.yaml",
                fetch=self._bucket({}),
            )

    @pytest.mark.asyncio
    async def test_a_non_list_include_fails_loudly(self):
        with pytest.raises(ValueError, match="'include' must be a list of file paths"):
            await resolve_bucket_includes(
                config={"include": "model_config.yaml"},
                object_key="config.yaml",
                fetch=self._bucket({}),
            )

    @pytest.mark.asyncio
    async def test_get_config_from_bucket_merges_includes_over_s3(self, monkeypatch):
        objects = {
            "lit6982/config.yaml": {
                "include": ["model_config.yaml"],
                "general_settings": {"master_key": MASTER_KEY},
            },
            "lit6982/model_config.yaml": {"model_list": [{"model_name": "included-model"}]},
        }
        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.s3_object_reader",
            lambda bucket_name: objects.get,
        )

        config = await get_config_from_bucket(
            bucket_type="s3", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert config == {
            "general_settings": {"master_key": MASTER_KEY},
            "model_list": [{"model_name": "included-model"}],
        }

    @pytest.mark.asyncio
    async def test_the_blocking_s3_work_runs_off_the_event_loop_thread(self, monkeypatch):
        loop_thread = threading.current_thread()
        threads = []

        def build_reader(bucket_name):
            threads.append(threading.current_thread())

            def read(object_key):
                threads.append(threading.current_thread())
                return {"model_list": [{"model_name": "a-model"}]}

            return read

        monkeypatch.setattr("litellm.proxy.common_utils.load_config_utils.s3_object_reader", build_reader)

        await get_config_from_bucket(bucket_type="s3", bucket_name="litellm-configs", object_key="config.yaml")

        assert len(threads) == 2 and loop_thread not in threads

    @pytest.mark.asyncio
    async def test_one_s3_client_serves_the_whole_include_tree(self, monkeypatch):
        objects = {
            "lit6982/config.yaml": {"include": ["model_config.yaml"]},
            "lit6982/model_config.yaml": {"model_list": [{"model_name": "included-model"}]},
        }
        readers = []

        def build_reader(bucket_name):
            requested = []
            readers.append(requested)

            def read(object_key):
                requested.append(object_key)
                return objects.get(object_key)

            return read

        monkeypatch.setattr("litellm.proxy.common_utils.load_config_utils.s3_object_reader", build_reader)

        await get_config_from_bucket(
            bucket_type="s3", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert readers == [["lit6982/config.yaml", "lit6982/model_config.yaml"]]

    @pytest.mark.asyncio
    async def test_an_empty_included_object_merges_as_an_empty_config(self, monkeypatch):
        objects = {
            "lit6982/config.yaml": "include:\n  - empty.yaml\nmodel_list:\n  - model_name: only-model\n",
            "lit6982/empty.yaml": "",
        }

        class FakeGCSBucket:
            async def download_gcs_object(self, object_key):
                return objects[object_key].encode("utf-8")

        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.gcs_config_bucket",
            lambda bucket_name: FakeGCSBucket(),
        )

        config = await get_config_from_bucket(
            bucket_type="gcs", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert config == {"model_list": [{"model_name": "only-model"}]}

    @pytest.mark.asyncio
    async def test_get_config_from_bucket_merges_includes_over_gcs(self, monkeypatch):
        objects = {
            "lit6982/config.yaml": {
                "include": ["model_config.yaml"],
                "general_settings": {"master_key": MASTER_KEY},
            },
            "lit6982/model_config.yaml": {"model_list": [{"model_name": "included-model"}]},
        }

        buckets = []

        class FakeGCSBucket:
            def __init__(self):
                self.requested = []
                buckets.append(self)

            async def download_gcs_object(self, object_key):
                self.requested.append(object_key)
                return yaml.dump(objects[object_key]).encode("utf-8")

        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.gcs_config_bucket",
            lambda bucket_name: FakeGCSBucket(),
        )

        config = await get_config_from_bucket(
            bucket_type="gcs", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert config == {
            "general_settings": {"master_key": MASTER_KEY},
            "model_list": [{"model_name": "included-model"}],
        }
        assert [bucket.requested for bucket in buckets] == [
            ["lit6982/config.yaml", "lit6982/model_config.yaml"]
        ]

    @pytest.mark.asyncio
    async def test_get_config_from_bucket_returns_none_when_the_root_object_is_missing(self, monkeypatch):
        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.s3_object_reader",
            lambda bucket_name: (lambda object_key: None),
        )

        assert (
            await get_config_from_bucket(
                bucket_type="s3", bucket_name="litellm-configs", object_key="missing.yaml"
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_an_object_pulled_in_twice_is_read_once(self):
        objects = {
            "configs/a.yaml": {"include": ["shared.yaml"]},
            "configs/b.yaml": {"include": ["./shared.yaml"]},
            "configs/shared.yaml": {"model_list": [{"model_name": "shared"}]},
        }
        requested = []

        async def fetch(object_key):
            requested.append(object_key)
            return objects.get(object_key)

        await resolve_bucket_includes(
            config={"include": ["a.yaml", "b.yaml"]},
            object_key="configs/config.yaml",
            fetch=fetch,
        )

        assert requested == ["configs/a.yaml", "configs/b.yaml", "configs/shared.yaml"]

    @pytest.mark.asyncio
    async def test_an_empty_root_object_does_not_boot_an_empty_proxy(self, monkeypatch):
        class FakeGCSBucket:
            async def download_gcs_object(self, object_key):
                return b""

        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.gcs_config_bucket",
            lambda bucket_name: FakeGCSBucket(),
        )

        config = await get_config_from_bucket(
            bucket_type="gcs", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert config is None

    @pytest.mark.asyncio
    async def test_an_object_that_is_not_valid_yaml_is_reported_as_a_yaml_error(self, monkeypatch, caplog):
        class FakeGCSBucket:
            async def download_gcs_object(self, object_key):
                return b"model_list: [\n"

        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.gcs_config_bucket",
            lambda bucket_name: FakeGCSBucket(),
        )

        with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
            config = await get_config_from_bucket(
                bucket_type="gcs", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
            )

        assert config is None
        assert [
            record
            for record in caplog.records
            if "not valid YAML" in record.getMessage() and "lit6982/config.yaml" in record.getMessage()
        ]


class TestGCSConfigBucketClient:
    @pytest.mark.asyncio
    async def test_reading_a_config_from_gcs_does_not_need_an_enterprise_license(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", False)

        bucket = gcs_config_bucket("litellm-configs")

        assert bucket is not None
        assert bucket.BUCKET_NAME == "litellm-configs"

    @pytest.mark.asyncio
    async def test_reading_a_config_from_gcs_starts_no_background_task(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
        running_before = asyncio.all_tasks()

        gcs_config_bucket("litellm-configs")

        assert asyncio.all_tasks() - running_before == set()
