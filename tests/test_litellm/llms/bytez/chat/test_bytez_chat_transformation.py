import os
import sys
import pytest
import json

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.bytez.chat.transformation import BytezChatConfig, API_BASE, version

TEST_API_KEY = "MOCK_BYTEZ_API_KEY"
TEST_MODEL_NAME = "google/gemma-3-4b-it"
TEST_MODEL = f"bytez/{TEST_MODEL_NAME}"
TEST_MESSAGES = [{"role": "user", "content": "Hello"}]


class TestBytezChatConfig:
    def test_validate_environment(self):
        config = BytezChatConfig()

        headers = {}

        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={},
            litellm_params={},
            api_key=TEST_API_KEY,
            api_base=API_BASE,
        )

        assert result["Authorization"] == f"Key {TEST_API_KEY}"
        assert result["content-type"] == "application/json"
        assert result["user-agent"] == f"litellm/{version}"

    def test_missing_api_key(self):
        with pytest.raises(Exception) as excinfo:
            config = BytezChatConfig()

            headers = {}

            config.validate_environment(
                headers=headers,
                model=TEST_MODEL,
                messages=TEST_MESSAGES,  # type: ignore
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=API_BASE,
            )

        assert "Missing api_key, make sure you pass in your api key" in str(
            excinfo.value
        )

    def test_bytez_completion_mock_sync(self, respx_mock):
        import litellm

        input_messages = [
            {"role": "user", "content": "What is your favorite kind of cat?"}
        ]

        output_content = "Hello, how can I help you today?"

        output = {
            "role": "assistant",
            "content": [{"type": "text", "text": output_content}],
        }

        # Mock the HTTP request
        respx_mock.post(f"{API_BASE}/{TEST_MODEL_NAME}").respond(
            json={
                "error": None,
                "output": output,
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = litellm.completion(
            model=TEST_MODEL,
            messages=input_messages,
            api_key=TEST_API_KEY,
            api_base=API_BASE,
        )

        assert response.choices[0].message.content == output_content  # type: ignore

    def test_bytez_messages_adaptation(self):
        cases = [
            dict(
                input=[
                    {
                        "role": "user",
                        "content": "What color is this cat?",
                    }
                ],
                expected_output=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this cat?"}
                        ],
                    }
                ],
            ),
            dict(
                input=[
                    {
                        "role": "user",
                        "content": {"type": "text", "text": "What color is this cat?"},
                    }
                ],
                expected_output=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this cat?"}
                        ],
                    }
                ],
            ),
            dict(
                input=[
                    {
                        "role": "user",
                        "content": [
                            "What color is this cat?",
                            {
                                "type": "image_url",
                                "url": "https://images.squarespace-cdn.com/content/v1/5452d441e4b0c188b51fef1a/1615326541809-TW01PVTOJ4PXQUXVRLHI/male-orange-tabby-cat.jpg",
                            },
                        ],
                    }
                ],
                expected_output=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this cat?"},
                            {
                                "type": "image",
                                "url": "https://images.squarespace-cdn.com/content/v1/5452d441e4b0c188b51fef1a/1615326541809-TW01PVTOJ4PXQUXVRLHI/male-orange-tabby-cat.jpg",
                            },
                        ],
                    }
                ],
            ),
            dict(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this cat?"},
                            {
                                "type": "image_url",
                                "url": "https://images.squarespace-cdn.com/content/v1/5452d441e4b0c188b51fef1a/1615326541809-TW01PVTOJ4PXQUXVRLHI/male-orange-tabby-cat.jpg",
                            },
                        ],
                    }
                ],
                expected_output=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this cat?"},
                            {
                                "type": "image",
                                "url": "https://images.squarespace-cdn.com/content/v1/5452d441e4b0c188b51fef1a/1615326541809-TW01PVTOJ4PXQUXVRLHI/male-orange-tabby-cat.jpg",
                            },
                        ],
                    }
                ],
            ),
            dict(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What kind of cat meow is this?"},
                            {
                                "type": "input_audio",
                                "url": "https://storage.googleapis.com/kagglesdsdata/datasets/1736753/2838478/dataset/dataset/B_ANI01_MC_FN_SIM01_101.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250711T192905Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=812b4bd6fcf9296f8e34f67664d900a81cf81a4c8a4f439ce12befc89b4bef07c2645cab20ce5ba8f6b311dffa85aa05b70b4efbe53bced50a43a5e7622ea1ee0d8cc390679cdc6a6aae2c27f75debc1ce2361c595b3c9e1b8c88e2756ffc6b4f290af7f3dfa7232dc69ccc9a2181be756e0d538250f9761a8b05ba1ac6c6b5d946f97a16aa14a5609ae62a2c4713c2077fcd34d129dbcdac6bb543ae547507b1a424e4fd09f817000943c11507e0a74c514ec212b17427b7fc9e2ce87a250db1258645e4862a4261e3790fd99c9186148ad0653acd2b6a9468adbeb94f17b5a685551037fd2cc9fe72fa405a006c0bd42d03be1e4c0dc4023ed3a77171edff3",
                            },
                        ],
                    }
                ],
                expected_output=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What kind of cat meow is this?"},
                            {
                                "type": "audio",
                                "url": "https://storage.googleapis.com/kagglesdsdata/datasets/1736753/2838478/dataset/dataset/B_ANI01_MC_FN_SIM01_101.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250711T192905Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=812b4bd6fcf9296f8e34f67664d900a81cf81a4c8a4f439ce12befc89b4bef07c2645cab20ce5ba8f6b311dffa85aa05b70b4efbe53bced50a43a5e7622ea1ee0d8cc390679cdc6a6aae2c27f75debc1ce2361c595b3c9e1b8c88e2756ffc6b4f290af7f3dfa7232dc69ccc9a2181be756e0d538250f9761a8b05ba1ac6c6b5d946f97a16aa14a5609ae62a2c4713c2077fcd34d129dbcdac6bb543ae547507b1a424e4fd09f817000943c11507e0a74c514ec212b17427b7fc9e2ce87a250db1258645e4862a4261e3790fd99c9186148ad0653acd2b6a9468adbeb94f17b5a685551037fd2cc9fe72fa405a006c0bd42d03be1e4c0dc4023ed3a77171edff3",
                            },
                        ],
                    }
                ],
            ),
            dict(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What kind of dog is this?"},
                            {
                                "type": "video_url",
                                "url": "https://storage.googleapis.com/kagglesdsdata/datasets/3957252/6888743/dog1.mp4?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250711T193025Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=810961d9abcbc2437954fdf19ef216deb65d3977eb354ec10af0d4644627cc6b143a5fc6450996bae1787c09d26334de7cd6ff887510a5ac2a6eed3cfcc6673a47686c84c1f2b0bf543009388d83f2cd9551ad5f72084513c6a7acd2c718849a4ebe951ccc5631bed014b0d115225c048b9f5de68673a37db24a98ad39cf3d0ba16fb764bf38eb90c78c295c21a4ddac08c3c661b65efd511ccb86bacb87a2e2a97a06f53ea1c64d5dcf274001a61bc20867802549601301d999f5a5b2e49fd444b7db860c68e1c67df6e8edd5ad97171eaafb4fa1462453924ea4d78733be411cb6b5c910d4f829cd7189c28dc1b22c8ae2a4da844a0d202e9e64bc7fb17947",
                            },
                        ],
                    }
                ],
                expected_output=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What kind of dog is this?"},
                            {
                                "type": "video",
                                "url": "https://storage.googleapis.com/kagglesdsdata/datasets/3957252/6888743/dog1.mp4?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250711T193025Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=810961d9abcbc2437954fdf19ef216deb65d3977eb354ec10af0d4644627cc6b143a5fc6450996bae1787c09d26334de7cd6ff887510a5ac2a6eed3cfcc6673a47686c84c1f2b0bf543009388d83f2cd9551ad5f72084513c6a7acd2c718849a4ebe951ccc5631bed014b0d115225c048b9f5de68673a37db24a98ad39cf3d0ba16fb764bf38eb90c78c295c21a4ddac08c3c661b65efd511ccb86bacb87a2e2a97a06f53ea1c64d5dcf274001a61bc20867802549601301d999f5a5b2e49fd444b7db860c68e1c67df6e8edd5ad97171eaafb4fa1462453924ea4d78733be411cb6b5c910d4f829cd7189c28dc1b22c8ae2a4da844a0d202e9e64bc7fb17947",
                            },
                        ],
                    }
                ],
            ),
        ]

        config = BytezChatConfig()

        for case in cases:
            messages = case["input"]
            expected_output = case["expected_output"]

            headers = {}

            headers = config.validate_environment(
                headers=headers,
                model=TEST_MODEL,
                messages=TEST_MESSAGES,  # type: ignore
                optional_params={},
                litellm_params={},
                api_key=TEST_API_KEY,
                api_base=API_BASE,
            )

            data = config.transform_request(
                model=TEST_MODEL_NAME,
                messages=messages,
                optional_params={},
                litellm_params={},
                headers=headers,
            )

            adapted_messages = data["messages"]
            stringified_output = json.dumps(adapted_messages)
            stringified_expected_output = json.dumps(expected_output)

            assert stringified_output == stringified_expected_output
