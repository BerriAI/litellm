"""
Integration tests for Bedrock S3 URL pass-through support.

These tests make real requests to AWS Bedrock and S3.
Requires AWS credentials and a pre-populated S3 bucket.

To run:
    AWS_PROFILE=<your-profile> BEDROCK_S3_TEST_BUCKET=<your-bucket> \
    pytest tests/local_testing/test_bedrock_s3_content.py -v

Required env vars (one of):
    AWS_PROFILE  — named profile with Bedrock + S3 access
    AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY

Required:
    BEDROCK_S3_TEST_BUCKET  — S3 bucket containing test files

Expected test files in the bucket:
    photos/cat.jpg, docs/report.pdf, videos/demo.mp4
"""

import os

import pytest

_has_aws_creds = bool(
    os.environ.get("AWS_PROFILE")
    or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
)
_s3_bucket = os.environ.get("BEDROCK_S3_TEST_BUCKET", "")
_skip_reason = "Requires AWS credentials (AWS_PROFILE or AWS_ACCESS_KEY_ID/SECRET) and BEDROCK_S3_TEST_BUCKET"


@pytest.fixture(autouse=True)
def _ensure_s3_input_flag(monkeypatch):
    """Patch supports_s3_input in factory.py for Nova models.

    Until this PR merges upstream, the remote model_prices JSON (fetched at
    import time) won't contain the supports_s3_input flag. We monkeypatch
    the function directly in factory.py where it's imported.
    """

    def _patched(model, custom_llm_provider=None):
        model_lower = model.lower()
        return (
            "nova" in model_lower
            and "micro" not in model_lower
            and "embed" not in model_lower
        )

    monkeypatch.setattr(
        "litellm.litellm_core_utils.prompt_templates.factory.supports_s3_input",
        _patched,
    )


@pytest.mark.skipif(not (_has_aws_creds and _s3_bucket), reason=_skip_reason)
class TestS3IntegrationBedrock:
    """Integration tests that send real requests to Bedrock with S3 content."""

    _MODELS = [
        "bedrock/us.amazon.nova-lite-v1:0",
        "bedrock/us.amazon.nova-pro-v1:0",
        "bedrock/us.amazon.nova-2-lite-v1:0",
    ]

    @pytest.mark.parametrize("model", _MODELS)
    def test_should_send_s3_image(self, model):
        import litellm

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"s3://{_s3_bucket}/photos/cat.jpg"},
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.parametrize("model", _MODELS)
    def test_should_send_s3_pdf(self, model):
        import litellm

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this document in one sentence."},
                        {
                            "type": "file",
                            "file": {"file_data": f"s3://{_s3_bucket}/docs/report.pdf"},
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.parametrize("model", _MODELS)
    def test_should_send_s3_video(self, model):
        import litellm

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video in one sentence."},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"s3://{_s3_bucket}/videos/demo.mp4"},
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.parametrize("model", _MODELS)
    def test_should_send_multiple_s3_files_in_one_message(self, model):
        import litellm

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what you see in these two items."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"s3://{_s3_bucket}/photos/cat.jpg"},
                        },
                        {
                            "type": "file",
                            "file": {"file_data": f"s3://{_s3_bucket}/docs/report.pdf"},
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_should_reject_s3_url_for_unsupported_model(self):
        """Claude models don't support s3Location — should get a clear error."""
        import litellm

        with pytest.raises(Exception, match="does not support s3://"):
            litellm.completion(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"s3://{_s3_bucket}/photos/cat.jpg"},
                            },
                        ],
                    }
                ],
                max_tokens=50,
            )
