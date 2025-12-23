"""
Test cross-user batch access permissions.

This test verifies that a batch and related files created by one API key
cannot be accessed, modified, or cancelled by a different API key.

Reference: https://github.com/BerriAI/litellm/pull/17401/files
"""

import time

import httpx
import openai
import pytest

from test_managed_files_base import (
    ManagedFilesBase,
    MODEL_NAMES,
    MODEL_IDS,
)


BATCH_ROUTES = [
    "/v1/files",
    "/files",
    "/v1/files/*",
    "/files/*",
    "/v1/batches",
    "/batches",
    "/v1/batches/*",
    "/batches/*",
]


class TestManagedFilesPermissions(ManagedFilesBase):
    """Test cases for cross-user batch access permissions.

    Verifies that:
    - User A can create and access their own batches and files
    - User B cannot access, retrieve, cancel, or delete User A's batches/files
    """

    master_api_key = "sk-1234"

    @classmethod
    def setup_class(cls):
        cls.admin_client = httpx.Client(base_url=cls.base_url, verify=False)

    @classmethod
    def teardown_class(cls):
        cls.admin_client.close()

    def user_suffix(self) -> str:
        return f"{time.strftime('%Y%m%d%H%M%S')}{int(time.time() * 1000) % 1000:03d}"

    def create_user_and_key(self, user_suffix: str) -> tuple[str, str]:
        user_email = f"test-user-{user_suffix}-{self.user_suffix()}@test.com"

        user_response = self.admin_client.post(
            "/user/new",
            json={
                "user_email": user_email,
                "user_alias": user_email,
                "user_role": "internal_user",
                "auto_create_key": "false",
            },
            headers={
                "Authorization": f"Bearer {self.master_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        assert user_response.status_code == 200, (
            f"Failed to create user: {user_response.status_code} - {user_response.text}"
        )
        user_data = user_response.json()
        user_id = user_data.get("user_id")

        key_response = self.admin_client.post(
            "/key/generate",
            json={
                "user_id": user_id,
                "key_alias": user_email,
                "allowed_routes": BATCH_ROUTES,
            },
            headers={
                "Authorization": f"Bearer {self.master_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        assert key_response.status_code == 200, (
            f"Failed to create key: {key_response.status_code} - {key_response.text}"
        )
        key_data = key_response.json()
        api_key = key_data.get("key")

        print(f"Created user {user_email} with key {key_data.get('key_alias')}")
        return user_id, api_key

    def create_user_key_and_client(
        self,
        user_suffix: str,
    ) -> tuple[str, str, openai.OpenAI]:
        user_id, api_key = self.create_user_and_key(user_suffix)
        return user_id, api_key, self.create_openai_client(api_key)

    def create_key_and_client(self, user_id: str, key_suffix: str) -> str:
        key_alias = f"additional-key-{key_suffix}-{self.user_suffix()}"
        key_response = self.admin_client.post(
            "/key/generate",
            json={
                "user_id": user_id,
                "key_alias": key_alias,
                "allowed_routes": BATCH_ROUTES,
            },
            headers={
                "Authorization": f"Bearer {self.master_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        assert key_response.status_code == 200, (
            f"Failed to create additional key: {key_response.status_code} - {key_response.text}"
        )
        api_key = key_response.json().get("key")
        print(f"Created additional key {api_key[:20]}... for user {user_id}")
        return api_key, self.create_openai_client(api_key)

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_retrieve_user_a_batch(self, tmp_path, model_name):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)
        batch = self.create_batch(client_A, batch_input_file.id)

        # User A retrieves their own batch
        batch_a = client_A.batches.retrieve(batch_id=batch.id)
        assert batch_a.id == batch.id, (
            "User A should be able to retrieve their own batch"
        )

        # User B cannot retrieve User A's batch
        try:
            client_B.batches.retrieve(batch_id=batch.id)
            pytest.fail("User B should NOT be able to retrieve User A's batch")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_cancel_user_a_batch(self, tmp_path, model_name):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)
        batch = self.create_batch(client_A, batch_input_file.id)

        # User B cannot cancel User A's batch
        try:
            client_B.batches.cancel(batch_id=batch.id)
            pytest.fail("User B should NOT be able to cancel User A's batch")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_retrieve_user_a_batch_input_file(self, tmp_path, model_name):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)

        # User A retrieves their own file
        file_a = client_A.files.retrieve(file_id=batch_input_file.id)
        assert file_a.id == batch_input_file.id, (
            "User A should be able to retrieve their own file"
        )

        # User B cannot retrieve User A's file
        try:
            client_B.files.retrieve(file_id=batch_input_file.id)
            pytest.fail("User B should NOT be able to retrieve User A's file")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_download_user_a_batch_input_file_content(
        self,
        tmp_path,
        model_name,
    ):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)

        # User A can download their own file content
        content_a = client_A.files.content(file_id=batch_input_file.id)
        assert content_a.text, (
            "User A should be able to download their own file content"
        )

        # User B cannot download User A's file content
        try:
            client_B.files.content(file_id=batch_input_file.id)
            pytest.fail("User B should NOT be able to download User A's file content")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_delete_user_a_batch_input_file(self, tmp_path, model_name):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)

        # User B cannot delete User A's file
        try:
            client_B.files.delete(file_id=batch_input_file.id)
            pytest.fail("User B should NOT be able to delete User A's file")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

        # User A can still retrieve their own file
        file_a = client_A.files.retrieve(file_id=batch_input_file.id)
        assert file_a.id == batch_input_file.id, "File should still exist"

        # User A can delete their own file
        try:
            client_A.files.delete(file_id=batch_input_file.id)
        except openai.APIError as e:
            pytest.fail(f"User A should be able to delete their own file: {e}")

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_retrieve_user_a_batch_output_file(
        self,
        tmp_path,
        model_name,
    ):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)
        batch = self.create_batch(client_A, batch_input_file.id)

        # Wait for batch to complete
        completed_batch = self.wait_for_batch_completed(client_A, batch.id)
        assert completed_batch.output_file_id, "Batch should have an output file"

        # User A retrieves their own output file
        file_a = client_A.files.retrieve(file_id=completed_batch.output_file_id)
        assert file_a.id == completed_batch.output_file_id, (
            "User A should be able to retrieve their own output file"
        )

        # User B cannot retrieve User A's output file
        try:
            client_B.files.retrieve(file_id=completed_batch.output_file_id)
            pytest.fail("User B should NOT be able to retrieve User A's output file")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_download_user_a_batch_output_file_content(
        self,
        tmp_path,
        model_name,
    ):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)
        batch = self.create_batch(client_A, batch_input_file.id)

        # Wait for batch to complete
        completed_batch = self.wait_for_batch_completed(client_A, batch.id)
        assert completed_batch.output_file_id, "Batch should have an output file"

        # User A can download their own output file content
        content_a = client_A.files.content(file_id=completed_batch.output_file_id)
        assert content_a.text, (
            "User A should be able to download their own output file content"
        )

        # User B cannot download User A's output file content
        try:
            client_B.files.content(file_id=completed_batch.output_file_id)
            pytest.fail(
                "User B should NOT be able to download User A's output file content",
            )
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_delete_user_a_batch_output_file(self, tmp_path, model_name):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)
        batch = self.create_batch(client_A, batch_input_file.id)

        # Wait for batch to complete
        completed_batch = self.wait_for_batch_completed(client_A, batch.id)
        assert completed_batch.output_file_id, "Batch should have an output file"

        # User B cannot delete User A's output file
        try:
            client_B.files.delete(file_id=completed_batch.output_file_id)
            pytest.fail("User B should NOT be able to delete User A's output file")
        except openai.PermissionDeniedError as e:
            assert e.status_code == 403

        # User A can still retrieve their own output file
        file_a = client_A.files.retrieve(file_id=completed_batch.output_file_id)
        assert file_a.id == completed_batch.output_file_id, (
            "Output file should still exist"
        )

        # User A can delete their own output file
        try:
            client_A.files.delete(file_id=completed_batch.output_file_id)
        except openai.APIError as e:
            pytest.fail(f"User A should be able to delete their own output file: {e}")

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_b_cannot_list_user_a_batches(self, tmp_path, model_name):
        user_a_id, user_a_key, client_A = self.create_user_key_and_client("a")
        user_b_id, user_b_key, client_B = self.create_user_key_and_client("b")

        # User A creates a batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_A, request_file)
        batch = self.create_batch(client_A, batch_input_file.id)

        # User A can see their own batch in the list
        batches_a = client_A.batches.list(
            limit=10,
            extra_query={"target_model_names": model_name},
        )
        batch_ids_a = [b.id for b in batches_a.data]
        assert batch.id in batch_ids_a, "User A should see their own batch in the list"

        # User B's batch list should NOT contain User A's batch
        batches_b = client_B.batches.list(
            limit=10,
            extra_query={"target_model_names": model_name},
        )
        batch_ids_b = [b.id for b in batches_b.data]
        assert batch.id not in batch_ids_b, (
            "User B should NOT see User A's batch in the list"
        )

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_user_api_keys_are_interchangeable(self, tmp_path, model_name):
        # Create user with 3 keys
        user_id, key1, client_Key1 = self.create_user_key_and_client("a")
        key2, client_Key2 = self.create_key_and_client(user_id, "a2")
        key3, client_Key3 = self.create_key_and_client(user_id, "a3")

        # Key1: Create batch input file and batch
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file = self.create_batch_input_file(client_Key1, request_file)
        batch = self.create_batch(client_Key1, batch_input_file.id)

        # Key1: Retrieve batch
        batch_retrieved = client_Key1.batches.retrieve(batch_id=batch.id)
        assert batch_retrieved.id == batch.id, "Key1 should retrieve its own batch"

        # Key2: Wait for batch completion and retrieve output
        completed_batch = self.wait_for_batch_completed(client_Key2, batch.id)
        assert completed_batch.output_file_id, "Batch should have an output file"

        # Key2: Retrieve output file metadata
        output_file = client_Key2.files.retrieve(file_id=completed_batch.output_file_id)
        assert output_file.id == completed_batch.output_file_id, (
            "Key2 should retrieve output file"
        )

        # Key2: Download output file content
        output_content = client_Key2.files.content(
            file_id=completed_batch.output_file_id,
        )
        assert output_content.text, "Key2 should download output file content"

        # Key3: List batches and verify batch is visible
        batches = client_Key3.batches.list(
            limit=10,
            extra_query={"target_model_names": model_name},
        )
        batch_ids = [b.id for b in batches.data]
        assert batch.id in batch_ids, "Key3 should see batch in list"

        # Key3: Delete input file
        try:
            client_Key3.files.delete(file_id=batch_input_file.id)
        except openai.APIError as e:
            pytest.fail(f"Key3 should delete input file: {e}")

        # Key3: Delete output file
        try:
            client_Key3.files.delete(file_id=completed_batch.output_file_id)
        except openai.APIError as e:
            pytest.fail(f"Key3 should delete output file: {e}")

        # Key1: Create another batch for cancellation test
        request_file2 = self.create_batch_request_file_on_disk(tmp_path, model_name)
        batch_input_file2 = self.create_batch_input_file(client_Key1, request_file2)
        batch2 = self.create_batch(client_Key1, batch_input_file2.id)

        # Key3: Cancel the batch created by Key1
        try:
            cancelled_batch = client_Key3.batches.cancel(batch_id=batch2.id)
            assert cancelled_batch.id == batch2.id, (
                "Key3 should cancel batch created by Key1"
            )
        except openai.BadRequestError:
            pass  # Batch may have already completed



